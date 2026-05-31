import torch
import numpy as np
import random
import pandas as pd
import os

from evaluation import run_evaluation

# Create results folder
os.makedirs("results", exist_ok=True)

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

num_runs = 5
all_results = []

for run in range(num_runs):
    seed = 42 + run
    print(f"\n===== RUN {run+1} | Seed = {seed} =====")

    set_seed(seed)

    metrics = run_evaluation()

    metrics["run"] = run + 1
    metrics["seed"] = seed

    all_results.append(metrics)

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Save all run results
results_df.to_csv("results/multi_run_results.csv", index=False)

# Compute summary only for scalar numeric metrics
summary_columns = [
    "accuracy",
    "ece",
    "brier",
    "coverage",
    "filtered_accuracy"
]

summary = results_df[summary_columns].agg(["mean", "std"])

# Save summary
summary.to_csv("results/final_summary.csv")

print("\n===== ALL RUNS =====")
print(results_df)

print("\n===== FINAL SUMMARY =====")
print(summary)
