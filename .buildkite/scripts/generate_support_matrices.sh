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
MODEL_CATEGORY=("text-only" "multimodel")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")
FEATURE_CATEGORY=("feature support matrix" "kernel support matrix" "quantization support matrix" "parallelism support matrix")

model_csv_files=()
feature_csv_files=()

# Output CSV files
model_support_matrix_csv="model_support_matrix.csv"
echo "Model,UnitTest,IntegrationTest,Benchmark" > "$model_support_matrix_csv"

feature_support_matrix_csv="feature_support_matrix.csv"
echo "Feature,CorrectnessTest,PerformanceTest" > "$feature_support_matrix_csv"

process_models_by_category() {
    local category="$1"
    local csv_filename="$2" # Pass filename in

    echo "Model,UnitTest,IntegrationTest,Benchmark" > "$csv_filename"

    # Loop through all models for this specific category
    for model in "${model_list[@]}"; do
        row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            # --- NEW KEY FORMAT ---
            # Get result using the new model:category:stage format
            result=$(buildkite-agent meta-data get "${model}:${category}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$csv_filename"
    done
}

process_features_by_category() {
    local category="$1"
    local csv_filename="$2"

    echo "Feature,CorrectnessTest,PerformanceTest" > "$csv_filename"

    for feature in "${feature_list[@]}"; do
        row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            # Get result using the new feature:category:stage format
            result=$(buildkite-agent meta-data get "${feature}:${category}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$csv_filename"
    done
}

# Loop through each category and generate its specific CSV
if [ ${#model_list[@]} -gt 0 ]; then
    for category in "${MODEL_CATEGORY[@]}"; do
        category_filename="$category"
        csv_filename="${category_filename}_model_support_matrix.csv"
        
        # Add to our list for later upload and cleanup
        model_csv_files+=("$csv_filename")

        echo "--- Generating matrix for category: $category ---"
        # Generate the CSV file for this category
        process_models_by_category "$category" "$csv_filename"
    done
fi

if [ ${#feature_list[@]} -gt 0 ]; then
    for category in "${FEATURE_CATEGORY[@]}"; do
        # Sanitize filename (e.g., "feature support matrix" -> "feature_support_matrix")
        category_filename=$(echo "$category" | tr ' ' '_')
        csv_filename="${category_filename}.csv"

        feature_csv_files+=("$csv_filename") # Add to list

        echo "--- Generating matrix for feature category: $category ---"
        process_features_by_category "$category" "$csv_filename"
    done
fi

buildkite-agent meta-data set "CI_TESTS_FAILED" "${ANY_FAILED}"

echo "--- Model support matrices (categorized) ---"
for csv_file in "${model_csv_files[@]}"; do
    echo "--- Matrix: $csv_file ---"
    cat "$csv_file"
done

echo "--- Feature support matrices (categorized) ---"
for csv_file in "${feature_csv_files[@]}"; do
    echo "--- Matrix: $csv_file ---"
    cat "$csv_file"
done

echo "--- Saving support matrices as Buildkite Artifacts ---"
for csv_file in "${model_csv_files[@]}"; do
    cat "$csv_file"
    buildkite-agent artifact upload "$csv_file"
done

for csv_file in "${feature_csv_files[@]}"; do
    buildkite-agent artifact upload "$csv_file"
done

echo "Reports uploaded successfully."

# cleanup
if [ ${#feature_csv_files[@]} -gt 0 ]; then
    rm "${feature_csv_files[@]}"
fi