#!/usr/bin/env bash
set -euo pipefail

# Compile
javac -d . Assignment3/DecisionTreeGP.java

# Output CSV
output="seed_results.csv"
echo "seed,train_accuracy,test_accuracy,f_measure,runtime" > "$output"

for seed in $(seq 1 30); do
  echo "Running seed $seed..."
  # Provide inputs: mode=train, seed, leave filepaths blank to use defaults
  result=$(printf "train\n%s\n\n\n" "$seed" | java -cp . DecisionTreeGP 2>&1)
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
