from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


DEFAULT_TEST_METRICS = {
    "completion_ratio": "higher",
    "disturbance_backlog_auc_mean": "lower",
    "disturbance_service_loss_auc_mean": "lower",
    "mean_service_ratio": "higher",
    "mean_backlog_to_capacity": "lower",
    "mean_raw_load_imbalance": "lower",
    "mean_kam_capacity_mismatch": "lower",
    "mean_makespan_proxy": "lower",
    "mean_cross_communication_ratio": "lower",
    "mean_coordination_cost_normalized": "lower",
}


def _holm_adjust(pvalues: List[float]) -> List[float]:
    if not pvalues:
        return []
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_method_comparisons(
    run_summary: pd.DataFrame,
    proposed=("c_kam", "d_kam"),
    metrics: Dict[str, str] = DEFAULT_TEST_METRICS,
) -> pd.DataFrame:
    """Paired tests over independent scenario seeds, never over time steps."""
    rows: List[Dict[str, object]] = []
    for scenario, sg in run_summary.groupby("scenario"):
        methods = sorted(set(sg["method"]))
        for candidate in proposed:
            if candidate not in methods:
                continue
            cg = sg[sg["method"] == candidate].set_index("seed")
            for baseline in methods:
                if baseline == candidate or baseline in proposed:
                    continue
                bg = sg[sg["method"] == baseline].set_index("seed")
                seeds = sorted(set(cg.index) & set(bg.index))
                if not seeds:
                    continue
                for metric, direction in metrics.items():
                    if metric not in cg or metric not in bg:
                        continue
                    a = cg.loc[seeds, metric].astype(float).to_numpy()
                    b = bg.loc[seeds, metric].astype(float).to_numpy()
                    improvement = a - b if direction == "higher" else b - a
                    sd = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0
                    dz = float(np.mean(improvement) / sd) if sd > 0 else 0.0
                    if len(seeds) > 1:
                        t_p = float(ttest_rel(a, b).pvalue)
                        try:
                            w_p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
                        except ValueError:
                            w_p = 1.0
                    else:
                        t_p = float("nan")
                        w_p = float("nan")
                    rows.append(
                        {
                            "scenario": scenario,
                            "proposed_method": candidate,
                            "baseline": baseline,
                            "metric": metric,
                            "direction": direction,
                            "n_pairs": len(seeds),
                            "mean_improvement": float(np.mean(improvement)),
                            "median_improvement": float(np.median(improvement)),
                            "sd_improvement": sd,
                            "effect_size_dz": dz,
                            "paired_t_p": t_p,
                            "wilcoxon_p": w_p,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["paired_t_p_holm"] = np.nan
    out["wilcoxon_p_holm"] = np.nan
    # Family: all proposed-vs-baseline contrasts for one scenario and outcome.
    for (_, _), idx in out.groupby(["scenario", "metric"]).groups.items():
        idx = list(idx)
        tvals = [float(out.loc[i, "paired_t_p"]) for i in idx]
        wvals = [float(out.loc[i, "wilcoxon_p"]) for i in idx]
        finite_t = [not np.isnan(v) for v in tvals]
        finite_w = [not np.isnan(v) for v in wvals]
        if all(finite_t):
            out.loc[idx, "paired_t_p_holm"] = _holm_adjust(tvals)
        if all(finite_w):
            out.loc[idx, "wilcoxon_p_holm"] = _holm_adjust(wvals)
    return out


def friedman_omnibus(
    run_summary: pd.DataFrame,
    metrics: Dict[str, str] = DEFAULT_TEST_METRICS,
) -> pd.DataFrame:
    """Friedman omnibus tests across methods for each scenario/outcome.

    Seeds are blocks. Kendall's W is reported as an omnibus effect size. These
    tests are only meaningful once enough independent seeds are used.
    """
    rows: List[Dict[str, object]] = []
    for scenario, sg in run_summary.groupby("scenario"):
        methods = sorted(sg["method"].unique().tolist())
        if len(methods) < 3:
            continue
        for metric, direction in metrics.items():
            if metric not in sg:
                continue
            pivot = sg.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            pivot = pivot.reindex(columns=methods).dropna()
            n, k = pivot.shape
            if n < 2 or k < 3:
                continue
            samples = [pivot[m].to_numpy(dtype=float) for m in methods]
            try:
                stat, p = friedmanchisquare(*samples)
            except ValueError:
                stat, p = 0.0, 1.0
            kendall_w = float(stat / max(n * (k - 1), 1))
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "direction": direction,
                    "n_blocks": int(n),
                    "n_methods": int(k),
                    "methods": ",".join(methods),
                    "friedman_chi2": float(stat),
                    "friedman_p": float(p),
                    "kendall_w": kendall_w,
                }
            )
    return pd.DataFrame(rows)


def ckam_dkam_comparison(
    run_summary: pd.DataFrame,
    metrics: Dict[str, str] = DEFAULT_TEST_METRICS,
) -> pd.DataFrame:
    """Paired D-KAM minus C-KAM comparison on identical seeds.

    Positive ``mean_improvement_dkam_vs_ckam`` always favors D-KAM according to
    the metric direction. The table is descriptive/inferential; neither method
    is designated as universally superior a priori.
    """
    rows: List[Dict[str, object]] = []
    for scenario, sg in run_summary.groupby("scenario"):
        c = sg[sg["method"] == "c_kam"].set_index("seed")
        d = sg[sg["method"] == "d_kam"].set_index("seed")
        seeds = sorted(set(c.index) & set(d.index))
        if not seeds:
            continue
        for metric, direction in metrics.items():
            if metric not in c or metric not in d:
                continue
            cv = c.loc[seeds, metric].astype(float).to_numpy()
            dv = d.loc[seeds, metric].astype(float).to_numpy()
            improvement = dv - cv if direction == "higher" else cv - dv
            sd = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0
            dz = float(np.mean(improvement) / sd) if sd > 0 else 0.0
            if len(seeds) > 1:
                t_p = float(ttest_rel(dv, cv).pvalue)
                try:
                    w_p = float(wilcoxon(dv, cv, zero_method="wilcox").pvalue)
                except ValueError:
                    w_p = 1.0
            else:
                t_p = float("nan")
                w_p = float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": len(seeds),
                    "mean_improvement_dkam_vs_ckam": float(np.mean(improvement)),
                    "median_improvement_dkam_vs_ckam": float(np.median(improvement)),
                    "effect_size_dz": dz,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One C-vs-D contrast per metric/scenario; correction across primary outcomes
    # is left to the final analysis plan rather than silently imposed here.
    return out
