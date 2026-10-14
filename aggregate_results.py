#!/usr/bin/env python3
"""Summarise repeated runs (several seeds and / or ablation variants) written by mainCode.ipynb.

    python aggregate_results.py --results ./results --dataset BUS

Reads <results>/<dataset>/<variant>/seed*/results.json and prints / saves:
  * mean +- std and 95 % t-confidence interval over seeds for accuracy, macro-F1 and Dice, per variant
  * one-way ANOVA across variants and Welch t-tests of every variant against `full`
Needs >= 2 seeds per variant for the statistics.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

METRICS = {"accuracy": ("classification", "accuracy"), "macro_f1": ("classification", "macro_f1"),
           "macro_auc": ("classification", "macro_auc"), "dice": ("segmentation", "dice_mean"), "iou": ("segmentation", "iou_mean")}


def load(results, dataset):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, dataset, "*", "seed*", "results.json"))):
        r = json.load(open(f)); parts = f.split(os.sep)
        row = dict(variant=parts[-3], seed=int(parts[-2].replace("seed", "")))
        for k, (a, b) in METRICS.items():
            row[k] = r.get(a, {}).get(b, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results"); ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    df = load(a.results, a.dataset)
    if df.empty:
        raise SystemExit("no results.json files found")
    out = []
    for v, g in df.groupby("variant"):
        for k in METRICS:
            x = g[k].dropna().values
            if len(x) == 0:
                continue
            sd = x.std(ddof=1) if len(x) > 1 else np.nan
            h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
            out.append(dict(variant=v, metric=k, n_runs=len(x), mean=x.mean(), std=sd, ci95_low=x.mean() - h, ci95_high=x.mean() + h))
    summ = pd.DataFrame(out)
    print(summ.round(4).to_string(index=False))
    summ.to_csv(os.path.join(a.results, a.dataset, "summary_over_seeds.csv"), index=False)

    tests = []
    for k in ("accuracy", "macro_f1", "dice"):
        groups = {v: g[k].dropna().values for v, g in df.groupby("variant")}
        groups = {v: x for v, x in groups.items() if len(x) >= 2}
        if len(groups) >= 2:
            F, p = stats.f_oneway(*groups.values())
            tests.append(dict(metric=k, test="one-way ANOVA", comparison=" vs ".join(groups), statistic=F, p_value=p))
        if "full" in groups:
            for v, x in groups.items():
                if v != "full":
                    t, p = stats.ttest_ind(groups["full"], x, equal_var=False)
                    tests.append(dict(metric=k, test="Welch t-test", comparison=f"full vs {v}", statistic=t, p_value=p))
    if tests:
        t = pd.DataFrame(tests); print("\n" + t.round(5).to_string(index=False))
        t.to_csv(os.path.join(a.results, a.dataset, "significance_tests.csv"), index=False)
    else:
        print("\n(statistical tests need >= 2 seeds for at least two variants)")


if __name__ == "__main__":
    main()
