#!/usr/bin/env bash
set -euo pipefail

# Compile
javac -d . Assignment3/DecisionTreeGP.java

# Output CSV
output="seed_results.csv"
echo "seed,train_accuracy,test_accuracy,f_measure,runtime" > "$output"

for seed in $(seq 1 30); do
  echo "Running seed $seed..."
  # Provide inputs: mode=train, seed, and explicit filepaths in Assignment3/
  result=$(printf "train\n%s\nAssignment3/Breast_train.csv\nAssignment3/Breast_test.csv\n" "$seed" | java -cp . DecisionTreeGP 2>&1)
  # Extract metrics
  train_acc=$(echo "$result" | grep -m1 "Training accuracy" | awk -F":" '{gsub("%","",$2); print $2}' | tr -d ' %')
  test_acc=$(echo "$result" | grep -m1 "Test accuracy" | awk -F":" '{gsub("%","",$2); print $2}' | tr -d ' %') || true
  f_measure=$(echo "$result" | grep -m1 "F-measure" | awk -F":" '{print $2}' | tr -d ' ') || true
  runtime=$(echo "$result" | grep -m1 "Runtime" | awk -F":" '{print $2}' | tr -d ' s') || true

  # Normalize empty fields
  train_acc=${train_acc:-0}
  test_acc=${test_acc:-0}
  f_measure=${f_measure:-0}
  runtime=${runtime:-0}

  echo "$seed,$train_acc,$test_acc,$f_measure,$runtime" >> "$output"
  echo "Seed $seed done. Test acc=$test_acc"
done

# Find best by test_accuracy, fallback to train_accuracy
best_seed=$(tail -n +2 "$output" | sort -t, -k3 -nr | head -n1 | cut -d, -f1)
best_line=$(grep "^$best_seed," "$output")

echo "\nBest seed: $best_seed"
echo "$best_line"

echo "Results written to $output"

# Re-run best seed to regenerate model text and capture final metrics
echo "Re-running best seed $best_seed to capture final model and metrics..."
printf "train\n%s\nAssignment3/Breast_train.csv\nAssignment3/Breast_test.csv\n" "$best_seed" | java -cp . DecisionTreeGP 2>&1 | tee best_run_output.log

# Extract best-run metrics
best_train_acc=$(grep -m1 "Training Accuracy" best_run_output.log | awk -F":" '{gsub("%","",$2); print $2}' | tr -d ' %')
best_test_acc=$(grep -m1 "Test Accuracy" best_run_output.log | awk -F":" '{gsub("%","",$2); print $2}' | tr -d ' %') || true
best_f=$(grep -m1 "F-measure" best_run_output.log | awk -F":" '{print $2}' | tr -d ' ') || true
best_runtime=$(grep -m1 "Runtime" best_run_output.log | awk -F":" '{print $2}' | tr -d ' s') || true

# Build result.txt
result_file="result.txt"
echo "Winning Seed Value: $best_seed" > "$result_file"
echo "" >> "$result_file"
echo "Final Classification Performance (Table 2):" >> "$result_file"
echo "Training Accuracy (%): ${best_train_acc}%" >> "$result_file"
echo "Test Accuracy (%): ${best_test_acc}%" >> "$result_file"
echo "F-measure: ${best_f}" >> "$result_file"
echo "Runtime: ${best_runtime} s" >> "$result_file"
echo "" >> "$result_file"
echo "Test accuracies for all 30 runs:" >> "$result_file"
tail -n +2 "$output" | awk -F, '{print $1 ": " $3 "%"}' >> "$result_file"

# Append the best model tree if available
if [ -f decision_tree_gp.model.txt ]; then
  echo "" >> "$result_file"
  echo "Best model tree (seed $best_seed):" >> "$result_file"
  echo "" >> "$result_file"
  cat decision_tree_gp.model.txt >> "$result_file"
fi

echo "Summary written to $result_file"
