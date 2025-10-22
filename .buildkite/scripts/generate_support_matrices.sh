#!/bin/bash
set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"

# Note: This script assumes the metadata keys contain newline-separated lists.
# The `mapfile` command reads these lists into arrays, correctly handling spaces.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")

# These arrays will hold the filenames of all generated CSVs
declare -a model_csv_files=()
declare -a feature_csv_files=()

process_models() {
    for model in "$@"; do
        # Get the category for this model, default to "text-only"
        local category
        category=$(buildkite-agent meta-data get "${model}_category" --default "text-only")
        # Define the category-specific CSV filename
        local category_filename=${category// /_}
        local category_csv="${category_filename}_model_support_matrix.csv"
        if [ ! -f "$category_csv" ]; then
            echo "Model,UnitTest,IntegrationTest,Benchmark" > "$category_csv"
            model_csv_files+=("$category_csv")
        fi
        # Build the row for the model
        local row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            local result
            result=$(buildkite-agent meta-data get "${model}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$category_csv"
    done
}

process_features() {
    for feature in "$@"; do
        # Get the category for this feature, default to "feature support matrix"
        local category
        category=$(buildkite-agent meta-data get "${feature}_category" --default "feature support matrix")
        # Define the category-specific CSV filename
        local category_filename=${category// /_}
        local category_csv="${category_filename}_feature_support_matrix.csv"
        if [ ! -f "$category_csv" ]; then
            echo "Feature,CorrectnessTest,PerformanceTest" > "$category_csv"
            feature_csv_files+=("$category_csv")
        fi
        # Build the row for the feature
        local row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            local result
            result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$category_csv"
    done
}

if [ ${#model_list[@]} -gt 0 ]; then
    process_models "${model_list[@]}"
fi

if [ ${#feature_list[@]} -gt 0 ]; then
    process_features "${feature_list[@]}"
fi

buildkite-agent meta-data set "CI_TESTS_FAILED" "${ANY_FAILED}"

echo "--- Model support matrices ---"
for csv_file in "${model_csv_files[@]}"; do
    echo "--- $csv_file ---"
    cat "$csv_file"
done

echo "--- Feature support matrices ---"
for csv_file in "${feature_csv_files[@]}"; do
    echo "--- $csv_file ---"
    cat "$csv_file"
done

echo "--- Saving support matrices as Buildkite Artifacts ---"
for csv_file in "${model_csv_files[@]}"; do
    buildkite-agent artifact upload "$csv_file"
done

for csv_file in "${feature_csv_files[@]}"; do
    buildkite-agent artifact upload "$csv_file"
done

echo "Reports uploaded successfully."

# cleanup
rm -f "${model_csv_files[@]}" "${feature_csv_files[@]}"