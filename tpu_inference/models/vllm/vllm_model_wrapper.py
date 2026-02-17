# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import inspect
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn
import torchax
import vllm.envs as vllm_envs
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import TORCH_DTYPE_TO_JAX
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.interfaces_base import is_pooling_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from tpu_inference.distributed.jax_parallel_state import \
    get_pp_group as jax_get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.logger import init_logger
from tpu_inference.models.common.interface import PoolerFunc
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)
from tpu_inference.runner.lora_utils import replace_lora_metadata

logger = init_logger(__name__)


class _VllmRunner(torch.nn.Module):

    def __init__(self, vllm_model: torch.nn.Module):
        super().__init__()
        self.vllm_model = vllm_model
        embed_input_ids_sig = inspect.signature(self.vllm_model.embed_input_ids)
        self._supports_is_multimodal = "is_multimodal" in embed_input_ids_sig.parameters
        # Deep stack models (e.g. Qwen3-VL) expect multimodal_embeddings as a
        # sequence of tensors for torch.cat, not a single tensor.
        self._needs_tuple_embeddings = hasattr(
            vllm_model, '_compute_deepstack_embeds')
        # Detect deepstack (Qwen3-VL, Qwen3-Omni, InternS1-Pro).
        self._has_deepstack = (
            getattr(vllm_model, 'use_deepstack', False)
            and getattr(vllm_model, 'deepstack_num_level', 0) > 0)
        self._deepstack_num_level = getattr(
            vllm_model, 'deepstack_num_level', 0)
        if self._has_deepstack:
            self._patch_deepstack_for_tpu()

        has_pooler = is_pooling_model(vllm_model)
        self.pooler = vllm_model.pooler if has_pooler else None

    def _patch_deepstack_for_tpu(self):
        """Patch deepstack methods for JIT compatibility.

        vLLM's deepstack uses mutable buffer side-channels
        (self.deepstack_input_embeds list with .copy_()) that are
        incompatible with @jax.jit. We replace:
        - _set_deepstack_input_embeds: captures tensor via attribute
          instead of .copy_() into pre-allocated buffers.
        - _clear_deepstack_input_embeds: no-op (buffer not used).

        This logic applies to every deepstack model (Qwen3 VL / Omni, Interns1-pro)
        """
        model = self.vllm_model
        model._tpu_captured_deepstack = None

        def _capture_set(deepstack_input_embeds):
            model._tpu_captured_deepstack = deepstack_input_embeds

        model._set_deepstack_input_embeds = _capture_set

        if hasattr(model, '_clear_deepstack_input_embeds'):
            model._clear_deepstack_input_embeds = lambda *args, **kwargs: None

    def forward(self, **kwargs) -> torch.Tensor:
        if "_dispatch" in kwargs:
            dispatch = kwargs.pop("_dispatch")
            if dispatch == "embed_multimodal":
                return self.embed_multimodal(**kwargs)
            elif dispatch == "embed_input_ids":
                return self.embed_input_ids(**kwargs)
        if "hidden_state" in kwargs:
            return self.compute_logits(kwargs["hidden_state"])
        else:
            return self.compute_hidden_state(
                kwargs["input_ids"],
                kwargs["positions"],
                kwargs["intermediate_tensors"],
                kwargs["inputs_embeds"],
            )

    def compute_hidden_state(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (self._has_deepstack
                and inputs_embeds is not None
                and inputs_embeds.dim() == 3):
            # inputs_embeds is (1 + num_level, num_tokens, hidden_size)
            # produced by concatenating [inputs_embeds, deepstack] in
            # _get_input_ids_embeds.  Split them back here.
            actual_embeds = inputs_embeds[0]     # (num_tokens, hidden_size)
            deepstack = inputs_embeds[1:]        # (num_level, num_tokens, H)

            num_levels = self._deepstack_num_level

            def _functional_get(num_tokens):
                return IntermediateTensors({
                    f"deepstack_input_embeds_{idx}":
                        deepstack[idx][:num_tokens]
                    for idx in range(num_levels)
                })

            orig_get = self.vllm_model._get_deepstack_input_embeds
            self.vllm_model._get_deepstack_input_embeds = _functional_get

            hidden_state = self.vllm_model(
                input_ids, positions, intermediate_tensors, actual_embeds)

            self.vllm_model._get_deepstack_input_embeds = orig_get
            return hidden_state

        hidden_state = self.vllm_model(input_ids, positions,
                                       intermediate_tensors, inputs_embeds)
        return hidden_state

    def compute_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.vllm_model.compute_logits(hidden_state)

    def embed_multimodal(self, **kwargs) -> Any:
        return self.vllm_model.embed_multimodal(**kwargs)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None,
                        **kwargs):
        if not self._supports_is_multimodal:
            kwargs.pop("is_multimodal", None)
        if self._needs_tuple_embeddings and multimodal_embeddings is not None:
            multimodal_embeddings = (multimodal_embeddings,)

        inputs_embeds = self.vllm_model.embed_input_ids(
            input_ids, multimodal_embeddings, **kwargs)

        if self._has_deepstack:
            # _set_deepstack_input_embeds was patched to capture here
            # instead of .copy_() into a buffer.
            deepstack = self.vllm_model._tpu_captured_deepstack
            self.vllm_model._tpu_captured_deepstack = None
            return inputs_embeds, deepstack

        return inputs_embeds, None


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh
    model: _VllmRunner

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config, self.mesh)
        self._apply_pp_patch()
        self._apply_mm_merge_patch()

    def _apply_pp_patch(self):
        # patch `get_pp_group` in vLLM to jax's get_pp_group.
        import sys

        import vllm.distributed as vllm_dist
        import vllm.distributed.parallel_state as vllm_ps

        vllm_ps.get_pp_group = jax_get_pp_group
        vllm_dist.get_pp_group = jax_get_pp_group

        for module_name, module in sys.modules.items():
            if module_name.startswith("vllm.model_executor.models"):
                if hasattr(module, "get_pp_group"):
                    setattr(module, "get_pp_group", jax_get_pp_group)

    def _apply_mm_merge_patch(self):
        """Patch vLLM's _merge_multimodal_embeddings for JIT compatibility.

        vLLM's version uses masked_scatter_ which torchax implements via
        jnp.nonzero — a data-dependent-shape op forbidden inside jax.jit.
        This replaces it with cumsum + gather + 3-arg where, matching the
        algorithm already used in tpu_inference's JAX multi_modal_utils.
        """
        import sys

        import vllm.model_executor.models.utils as vllm_model_utils

        _flatten = vllm_model_utils._flatten_embeddings

        def _jit_merge_multimodal_embeddings(inputs_embeds,
                                             multimodal_embeddings,
                                             is_multimodal):
            if len(multimodal_embeddings) == 0:
                return inputs_embeds

            mm_embeds_flat = _flatten(multimodal_embeddings)

            # Build padded array: [dummy_row, emb_0, emb_1, ..., emb_N]
            # so that index 0 maps to a zero row (for non-multimodal tokens)
            # and cumsum of is_multimodal gives 1-based indices into the
            # real embeddings.
            dummy_row = torch.zeros_like(mm_embeds_flat[0:1])
            padded = torch.cat([dummy_row, mm_embeds_flat], dim=0)

            gather_indices = torch.cumsum(is_multimodal.to(torch.int32),
                                          dim=0)
            update_values = padded[gather_indices]

            return torch.where(
                is_multimodal.unsqueeze(-1),
                update_values.to(dtype=inputs_embeds.dtype),
                inputs_embeds,
            )

        # Patch at the source module so future imports pick it up.
        vllm_model_utils._merge_multimodal_embeddings = (
            _jit_merge_multimodal_embeddings)

        # Safety net: also patch any model modules that already imported it.
        for module_name, module in sys.modules.items():
            if module_name.startswith("vllm.model_executor.models"):
                if hasattr(module, "_merge_multimodal_embeddings"):
                    setattr(module, "_merge_multimodal_embeddings",
                            _jit_merge_multimodal_embeddings)

    def load_weights(self):
        loading_start = time.time()
        # Set up to load the model into CPU first.
        # Cache device slice config since device config cannot be deepcopied
        modified_slice_config = False
        if hasattr(
                self.vllm_config.device_config,
                'slice') and self.vllm_config.device_config.slice is not None:
            slice_config = self.vllm_config.device_config.slice
            modified_slice_config = True
            self.vllm_config.device_config.slice = None
        self.vllm_config.compilation_config.static_forward_context.clear()

        vllm_config_for_load = copy.deepcopy(self.vllm_config)
        if modified_slice_config:
            self.vllm_config.device_config.slice = slice_config
        assert self.vllm_config.model_config.dtype in TORCH_DTYPE_TO_JAX, "The model_config.dtype must be a PyTorch dtype."
        vllm_config_for_load.device_config.device = "cpu"
        # Remove the dynamically added sharding_config attribute to avoid errors
        # when vLLM's replace() function checks for dataclass fields.
        # This is safe because vllm_config_for_load is only used for model loading
        # which doesn't need sharding_config, and self.vllm_config still has it.
        if hasattr(vllm_config_for_load, 'sharding_config'):
            delattr(vllm_config_for_load, 'sharding_config')
        # Clearing the cached compilation config, otherwise vllm model init will fail

        # When expert parallelism is enabled, vLLM loads weight in sharding
        # aware manner. Since tpu-inference has its own sharding logic, this
        # may casue errors. Therefore, we disable it during weight loading.
        vllm_config_for_load.parallel_config.enable_expert_parallel = False

        use_random_weights = (
            vllm_config_for_load.load_config.load_format == "dummy")
        if use_random_weights:
            logger.info(
                "Initializing vLLM model with random weights, weight loading skipped."
            )
        # The DummyModelLoader in vLLM calls torch._sync for torch_xla path when
        # it detects the tpu platform, but we don't need it and it causes crash
        # without proper setup.
        load_context = patch(
            "torch._sync",
            return_value=None) if use_random_weights else nullcontext()

        # By default load weights to the CPU device first. If we are running
        # under Pathways, this would cause weights to be loaded on a CPU-only
        # node, so we'll need to remove this context.
        jax_context = jax.default_device(
            jax.devices("cpu")
            [0]) if not vllm_envs.VLLM_TPU_USING_PATHWAYS else nullcontext()

        # Load the vLLM model and wrap it into a new model whose forward
        # function can calculate the hidden_state and logits.
        with load_context, jax_context:
            vllm_model = vllm_get_model(vllm_config=vllm_config_for_load)
        lora_manager = None
        if vllm_config_for_load.lora_config is not None:
            # Replace layers in the model with LoRA layers.
            with torchax.default_env():
                # Argument "device" in load_lora_model is used to set the device
                # used in punica wrapper.
                lora_manager, vllm_model = load_lora_model(
                    vllm_model, vllm_config_for_load, device="jax")
            replace_set_lora(vllm_model)

        static_forward_context = vllm_config_for_load.compilation_config.static_forward_context
        self.vllm_config.compilation_config.static_forward_context = static_forward_context
        self.vllm_config.compilation_config.static_all_moe_layers = vllm_config_for_load.compilation_config.static_all_moe_layers

        self.model = _VllmRunner(vllm_model)
        params_and_buffers = shard_model_to_tpu(self.model, self.mesh)

        self._pooler: Pooler | None = self.model.pooler

        loading_end = time.time()
        total_loading_time = loading_end - loading_start
        logger.info(
            f"Total time to load model weights from storage to TPU: {total_loading_time:.2f} seconds."
        )
        # Returning to the jax land, so we need to wrap it into a JaxValue.
        return jax_view(params_and_buffers), lora_manager

    def jit_step_func(self):

        @functools.partial(
            jax.jit,
            donate_argnames=("kv_caches", ),
            out_shardings=(
                None,  # kv_caches - keep original sharding
                NamedSharding(self.mesh,
                              PartitionSpec(ShardingAxisName.ATTN_DATA, None)),
                None,  # empty list
            ),
            compiler_options={
                "xla_tpu_all_gather_collective_matmul_mode":
                "post_spmd_conservative",
                "xla_tpu_reduce_scatter_collective_matmul_mode":
                "post_spmd_conservative"
            },
            static_argnames=("layer_name_to_kvcache_index", "is_first_rank",
                             "is_last_rank"),
        )
        def step_fun(
            params_and_buffers,  # This has been wrapped into torchax TorchValue
            kv_caches: List[jax.Array],
            input_ids: jax.Array,
            attn_metadata: AttentionMetadata,
            input_embeds: jax.Array,
            input_positions: jax.Array,
            layer_name_to_kvcache_index: Sequence[Tuple[str, int]],
            lora_metadata,
            intermediate_tensors: JaxIntermediateTensors = None,
            is_first_rank: bool = True,
            is_last_rank: bool = True,
            *args,
        ) -> Tuple[List[jax.Array], jax.Array]:
            layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
            lora_metadata = torch_view(lora_metadata)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=kv_caches,
                    mesh=self.mesh,
                    layer_name_to_kvcache_index=layer_name_to_kvcache_index
            ), set_forward_context(attn_metadata=attn_metadata,
                                   vllm_config=self.vllm_config):
                # We need to wrap args from jax land into TorchValue with
                # torch_view in order to call the Torch function.
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                if not is_first_rank:
                    intermediate_tensors = intermediate_tensors.to_torch()
                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": None if input_embeds is not None else torch_view(input_ids),
                        "positions": torch_view(input_positions),
                        "intermediate_tensors": intermediate_tensors,
                        "inputs_embeds": torch_view(input_embeds) if input_embeds is not None else None,
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
            # Wrap the output(hidden states or intermediate tensor)
            # from torch land into a JaxValue for the jax code to consume.
            if not is_last_rank:
                output = JaxIntermediateTensors.from_torch(output_from_torch)
            else:
                output = jax_view(output_from_torch)
            return new_kv_caches, output, []

        return step_fun

    def jit_compute_logits_func(self):

        @functools.partial(
            jax.jit,
            out_shardings=(NamedSharding(
                self.mesh,
                PartitionSpec(ShardingAxisName.MLP_DATA,
                              ShardingAxisName.MLP_TENSOR))),
        )
        def compute_logits_func(
            params_and_buffers: Any,
            hidden_states: jax.Array,
            lora_metadata,
        ) -> jax.Array:
            lora_metadata = torch_view(lora_metadata)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=None, mesh=self.mesh):
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                logits = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "hidden_state": torch_view(hidden_states),
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
            return jax_view(logits)

        return compute_logits_func

    def make_embed_multimodal_fn(self):

        def embed_multimodal_fn(state, image_grid_thw,
                                **kwargs) -> tuple[jax.Array, ...]:
            converted_kwargs = {
                k: torch_view(jnp.array(v)) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            # Convert tuple back to torchax tensor for vllm models that assert grid_thw.ndim == 2. The tuple form is used upstream for JIT hashability.
            # This might be required change in the vllm, as grid_thw does not really need to be tensors.
            if isinstance(image_grid_thw, tuple) and image_grid_thw:
                image_grid_thw = torch_view(
                    jnp.array(image_grid_thw, dtype=jnp.int32))
            with torchax.default_env(), set_forward_context(
                    attn_metadata=None, vllm_config=self.vllm_config):
                output = torch.func.functional_call(
                    self.model,
                    torch_view(state),
                    kwargs={
                        "_dispatch": "embed_multimodal",
                        "image_grid_thw": image_grid_thw,
                        **converted_kwargs,
                    },
                    tie_weights=False,
                )

            if isinstance(output, (list, tuple)):
                return tuple(jax_view(t) for t in output)
            return (jax_view(output), )

        return embed_multimodal_fn

    def make_embed_input_ids_fn(self):

        @jax.jit
        def embed_input_ids_fn(state, input_ids,
                               multimodal_embeddings,
                               is_multimodal):
            with torchax.default_env():
                output = torch.func.functional_call(
                    self.model,
                    torch_view(state),
                    kwargs={
                        "_dispatch": "embed_input_ids",
                        "input_ids": torch_view(input_ids),
                        "multimodal_embeddings": torch_view(
                            multimodal_embeddings)
                        if multimodal_embeddings is not None else None,
                        "is_multimodal": torch_view(is_multimodal)
                        if is_multimodal is not None else None,
                    },
                    tie_weights=False,
                )
            # _VllmRunner.embed_input_ids always returns
            # (inputs_embeds, deepstack_or_None).
            inputs_embeds, deepstack = output
            jax_embeds = jax_view(inputs_embeds)
            jax_deepstack = (
                jax_view(deepstack) if deepstack is not None else None)
            return jax_embeds, jax_deepstack

        return embed_input_ids_fn

    def make_get_mrope_input_positions_fn(self):
        if not hasattr(self.model.vllm_model, "get_mrope_input_positions"):
            return None

        get_mrope_input_positions = self.model.vllm_model.get_mrope_input_positions

        def _to_jax_array(value: Any) -> jax.Array:
            if isinstance(value, jax.Array):
                return value
            if isinstance(value, torch.Tensor):
                # jax_view failed to deal with torch.Tensor properly.
                return jnp.array(value.detach().cpu().numpy())
            return jnp.array(value)

        def get_mrope_input_positions_fn(*args, **kwargs):
            if args:
                input_tokens = args[0]
                mm_features = args[1] if len(args) > 1 else kwargs.pop(
                    "mm_features", None)
            else:
                input_tokens = kwargs.pop("input_tokens")
                mm_features = kwargs.pop("mm_features", None)

            positions, mrope_position_delta = get_mrope_input_positions(
                input_tokens, mm_features)
            return _to_jax_array(positions), _to_jax_array(
                mrope_position_delta)

        return get_mrope_input_positions_fn

    def build_pooler_func(self) -> PoolerFunc:

        def compute_pooler_output(
            hidden_states: jax.Array,
            pooling_metadata: PoolingMetadata,
            seq_lens: np.ndarray,
        ) -> PoolerOutput:
            assert self._pooler is not None, "Model does not support pooling"

            torch_states: torch.Tensor = torch_view(hidden_states)
            with torchax.default_env():
                torch_states = torch_states.to('cpu', non_blocking=True)
                pooling_metadata.build_pooling_cursor(
                    seq_lens,
                    torch.tensor(seq_lens),
                    device=torch_states.device,
                )
                outputs: list[torch.Tensor] = self._pooler(
                    torch_states,
                    pooling_metadata,
                )
                return outputs

        return compute_pooler_output


def load_lora_model(model: torch.nn.Module, vllm_config: VllmConfig,
                    device: str) -> torch.nn.Module:
    if not supports_lora(model):
        raise ValueError(
            f"{model.__class__.__name__} does not support LoRA yet.")

    if supports_multimodal(model):
        logger.warning("Regarding multimodal models, vLLM currently "
                       "only supports adding LoRA to language model.")

    # Add LoRA Manager to the Model Runner
    lora_manager = LRUCacheWorkerLoRAManager(
        vllm_config,
        device,
        model.embedding_modules,
    )
    return lora_manager, lora_manager.create_lora_manager(model)


# The reason why replace the method is that the set_lora and reset_lora need to
# run under torchax env.
def replace_set_lora(model):

    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        with torchax.default_env():
            self._original_set_lora(index, lora_a, lora_b)

    def _tpu_reset_lora(self, index: int):
        with torchax.default_env():
            self._original_reset_lora(index)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(
                module, module.__class__)
