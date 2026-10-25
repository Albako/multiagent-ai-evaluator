import json
from collections import defaultdict
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("analysis_output")


def safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 2) if clean else 0.0


def safe_stdev(values: list) -> float:
    clean = [v for v in values if v is not None]
    return round(statistics.stdev(clean), 2) if len(clean) > 1 else 0.0


def extract_node_latencies_from_trace(trace: list[dict]) -> dict[str, list[float]]:
    """Extracts individual node latencies (ms) for PC1, PC2, PC3, and PC4 from execution_trace."""
    latencies = defaultdict(list)
    for step in trace:
        comp = step.get("component", "")
        details = step.get("details", {})
        ms = details.get("latency_ms")
        if ms is None:
            continue
        if "PC1" in comp or comp == "Worker1":
            latencies["PC1"].append(float(ms))
        elif "PC2" in comp or comp == "Worker2":
            latencies["PC2"].append(float(ms))
        elif "PC3" in comp or comp == "Worker3":
            latencies["PC3"].append(float(ms))
        elif "PC4" in comp or "Judge" in comp:
            latencies["PC4"].append(float(ms))
    return latencies


def analyze_coding_results(coding_path: Path) -> dict:
    with open(coding_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_run = defaultdict(list)
    by_run_cat = defaultdict(list)
    node_ms_global = defaultdict(list)
    straggler_counts = defaultdict(int)
    parallel_saved_ms = []

    for r in records:
        run_name = r["run_name"]
        cat = r["category"]
        by_run[run_name].append(r)
        by_run_cat[(run_name, cat)].append(r)

        trace = r.get("execution_trace", [])
        for node, ms_list in extract_node_latencies_from_trace(trace).items():
            node_ms_global[node].extend(ms_list)

        for step in trace:
            if step.get("component") == "Parallel_Stage_Summary":
                det = step.get("details", {})
                if det.get("straggler_node"):
                    straggler_counts[det["straggler_node"]] += 1
                if det.get("time_saved_ms") is not None:
                    parallel_saved_ms.append(float(det["time_saved_ms"]))

    summary_rows = []
    for run_name, items in by_run.items():
        pass_rate = safe_mean([100.0 if x.get("unit_test_passed") else 0.0 for x in items])
        poisoned_items = [x for x in items if x.get("is_poisoned_15pct")]
        caught_count = sum(1 for x in poisoned_items if x.get("judge_caught_injection") is True)
        catch_rate = round((caught_count / len(poisoned_items)) * 100.0, 1) if poisoned_items else 0.0

        summary_rows.append({
            "Run": run_name,
            "N": len(items),
            "Pass_Rate_Pct": pass_rate,
            "Mean_Tokens": safe_mean([x.get("total_tokens", 0) for x in items]),
            "Prompt_Tokens": safe_mean([x.get("prompt_tokens", 0) for x in items]),
            "Completion_Tokens": safe_mean([x.get("completion_tokens", 0) for x in items]),
            "Mean_Latency_ms": safe_mean([x.get("total_latency_ms", 0) for x in items]),
            "Stdev_Latency_ms": safe_stdev([x.get("total_latency_ms", 0) for x in items]),
            "Mean_CWE_Warnings": safe_mean([x.get("static_cwe_count", 0) for x in items]),
            "Injection_15pct_Caught_Pct": catch_rate
        })

    cat_rows = []
    for (run_name, cat), items in sorted(by_run_cat.items()):
        cat_rows.append({
            "Run": run_name,
            "Category": cat,
            "N": len(items),
            "Pass_Rate_Pct": safe_mean([100.0 if x.get("unit_test_passed") else 0.0 for x in items]),
            "Mean_Tokens": safe_mean([x.get("total_tokens", 0) for x in items]),
            "Mean_Latency_ms": safe_mean([x.get("total_latency_ms", 0) for x in items]),
            "Mean_CWE_Warnings": safe_mean([x.get("static_cwe_count", 0) for x in items]),
        })

    return {
        "raw": records,
        "summary_rows": summary_rows,
        "cat_rows": cat_rows,
        "node_ms_global": node_ms_global,
        "straggler_counts": dict(straggler_counts),
        "mean_parallel_saved_ms": safe_mean(parallel_saved_ms)
    }


def analyze_text_results(text_path: Path) -> dict:
    with open(text_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_run = defaultdict(list)
    by_run_cat = defaultdict(list)

    for r in records:
        by_run[r["run_name"]].append(r)
        by_run_cat[(r["run_name"], r["category"])].append(r)

    summary_rows = []
    for run_name, items in by_run.items():
        poisoned_items = [x for x in items if x.get("is_poisoned_15pct")]
        caught_count = sum(1 for x in poisoned_items if x.get("judge_caught_injection") is True)
        catch_rate = round((caught_count / len(poisoned_items)) * 100.0, 1) if poisoned_items else 0.0

        summary_rows.append({
            "Run": run_name,
            "N": len(items),
            "Mean_Judge_Factual_Score": safe_mean([x.get("judge_factual_score") for x in items]),
            "Mean_Tokens": safe_mean([x.get("total_tokens", 0) for x in items]),
            "Mean_Latency_ms": safe_mean([x.get("total_latency_ms", 0) for x in items]),
            "Injection_15pct_Caught_Pct": catch_rate
        })

    return {"raw": records, "summary_rows": summary_rows, "by_run_cat": by_run_cat}


def rows_to_markdown_table(rows: list[dict], title: str) -> str:
    if not rows:
        return f"### {title}\n_No data available._\n"
    headers = list(rows[0].keys())
    lines = [
        f"### {title}",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def plot_charts(coding_data: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cat_rows = coding_data["cat_rows"]
    raw_records = coding_data["raw"]

    # --- CHART 1: Pareto Curve (Token Budget vs Pass Rate by Category) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    markers = {"coding_easy": "o", "coding_optimization": "s", "coding_security": "^"}
    colors = {
        "Fast_Coding": "#2b5c8f",
        "MultiAgent_Coding_Parallel": "#d95f02",
        "MultiAgent_Coding_Sequential": "#1b9e77"
    }

    for r in cat_rows:
        run = r["Run"]
        cat = r["Category"]
        ax.scatter(
            r["Mean_Tokens"],
            r["Pass_Rate_Pct"],
            s=110,
            color=colors.get(run, "#333333"),
            marker=markers.get(cat, "o"),
            label=f"{run} [{cat.replace('coding_', '')}]"
        )
        ax.annotate(
            f"{r['Pass_Rate_Pct']:.0f}%",
            (r["Mean_Tokens"], r["Pass_Rate_Pct"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8
        )

    ax.set_title("Pareto Efficiency: Prompt Budget (Tokens) vs. Unit Test Pass Rate")
    ax.set_xlabel("Mean Prompt Budget (Total Tokens per Task)")
    ax.set_ylabel("Unit Test Pass Rate (%)")
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart1_pareto_token_budget.png")
    plt.close(fig)

    # --- CHART 2: Pass Rate by Task Complexity Stratum ---
    categories = ["coding_easy", "coding_optimization", "coding_security"]
    runs = sorted({r["Run"] for r in cat_rows})
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for idx, run in enumerate(runs):
        vals = [
            next((r["Pass_Rate_Pct"] for r in cat_rows if r["Run"] == run and r["Category"] == c), 0.0)
            for c in categories
        ]
        rects = ax.bar(x + (idx - len(runs) / 2 + 0.5) * width, vals, width, label=run, color=colors.get(run))
        ax.bar_label(rects, fmt="%.1f%%", padding=3, fontsize=8)

    ax.set_title("Unit Test Pass Rate Across Task Complexity Strata")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy (MBPP+)", "Optimization (HumanEval+)", "Security (SecurityEval)"])
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart2_coding_complexity_bars.png")
    plt.close(fig)

    # --- CHART 3: Hardware Asymmetry & Node Latencies (PC1-PC4) ---
    node_ms = coding_data["node_ms_global"]
    if node_ms:
        nodes = sorted(node_ms.keys())
        means = [safe_mean(node_ms[n]) for n in nodes]
        stdevs = [safe_stdev(node_ms[n]) for n in nodes]

        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
        bars = ax.bar(nodes, means, yerr=stdevs, capsize=5, color=["#4c72b0", "#c44e52", "#55a868", "#8172b2"])
        ax.bar_label(bars, fmt="%.0f ms", padding=4, fontsize=8.5)
        ax.set_title("Inference Latency Across Asymmetric Hardware Nodes (PC1–PC4)")
        ax.set_ylabel("Mean Node Latency (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "chart3_hardware_asymmetry_latencies.png")
        plt.close(fig)

    # --- CHART 4: 15% Failure Injection Resilience ---
    runs_all = sorted({r["run_name"] for r in raw_records})
    propagated_pct = []
    caught_pct = []

    for run in runs_all:
        poisoned = [r for r in raw_records if r["run_name"] == run and r.get("is_poisoned_15pct")]
        if not poisoned:
            propagated_pct.append(0.0)
            caught_pct.append(0.0)
            continue
        caught = sum(1 for r in poisoned if r.get("judge_caught_injection") is True)
        c_rate = (caught / len(poisoned)) * 100.0
        caught_pct.append(c_rate)
        propagated_pct.append(100.0 - c_rate)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x_pos = np.arange(len(runs_all))
    ax.bar(x_pos, caught_pct, width=0.45, label="Caught & Blocked by Judge (REJECT_*)", color="#2ca02c")
    ax.bar(x_pos, propagated_pct, width=0.45, bottom=caught_pct, label="Uncaught Error Propagated", color="#d62728")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(runs_all, rotation=12)
    ax.set_ylabel("Share of Poisoned Tasks (15% Injection Rate) [%]")
    ax.set_ylim(0, 115)
    ax.set_title("Resilience to 15% Failure Injection: Single-Agent vs. Multi-Agent")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart4_failure_injection_15pct.png")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_md = ["# Statistic Benchmark Raport\n"]

    coding_file = Path("results_30_coding.json")
    if coding_file.exists():
        c_data = analyze_coding_results(coding_file)
        report_md.append(rows_to_markdown_table(c_data["summary_rows"], "Table 1: Results for 30 coding excersises"))
        report_md.append(rows_to_markdown_table(c_data["cat_rows"], "Table 2: Results for coding excersises per layers of complexity"))

        node_rows = [
            {
                "Node": node,
                "Calls_Count": len(ms_list),
                "Mean_Latency_ms": safe_mean(ms_list),
                "Stdev_Latency_ms": safe_stdev(ms_list),
                "Max_Latency_ms": round(max(ms_list), 2) if ms_list else 0.0
            }
            for node, ms_list in sorted(c_data["node_ms_global"].items())
        ]
        report_md.append(rows_to_markdown_table(node_rows, "Table 3: Latency per node"))
        report_md.append(f"**Bottleneck (Straggler Node counts in Parallel mode):** `{json.dumps(c_data['straggler_counts'])}`\n")
        report_md.append(f"**Average time saved by parallelization (`time_saved_ms`):** `{c_data['mean_parallel_saved_ms']} ms`\n")

        plot_charts(c_data)

    text_file = Path("results_50_text.json")
    if text_file.exists():
        t_data = analyze_text_results(text_file)
        report_md.append(rows_to_markdown_table(t_data["summary_rows"], "Table 4: Results for 50 text excersises"))

    out_md_path = OUTPUT_DIR / "tables_summary.md"
    out_md_path.write_text("\n".join(report_md), encoding="utf-8")
    print(f"Analysis complete! Tables and charts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
