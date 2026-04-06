import csv
import os
from collections import Counter, defaultdict


def _align_chars(gt: str, pred: str) -> list[tuple[str, str]]:
    """Character-level alignment with simple dynamic programming."""
    n, m = len(gt), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            candidates = [
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
                (dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"),
            ]
            best_cost, op = min(candidates, key=lambda x: x[0])
            dp[i][j] = best_cost
            back[i][j] = op

    aligned: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"M", "S"}:
            aligned.append((gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            aligned.append((gt[i - 1], "<del>"))
            i -= 1
        else:
            aligned.append(("<ins>", pred[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def run_error_analysis(
    per_sample_csv: str = "reports/benchmark_per_sample.csv",
    top_failures_csv: str = "reports/error_top_failures.csv",
    confusion_csv: str = "reports/error_char_confusions.csv",
    summary_csv: str = "reports/error_summary_by_engine.csv",
    top_k: int = 200,
) -> None:
    if not os.path.exists(per_sample_csv):
        raise FileNotFoundError(f"Per-sample benchmark file not found: {per_sample_csv}")

    rows: list[dict] = []
    with open(per_sample_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["success"] = int(row.get("success", "0") or 0)
            try:
                row["cer"] = float(row.get("cer", "0") or 0)
            except ValueError:
                row["cer"] = 0.0
            try:
                row["wer"] = float(row.get("wer", "0") or 0)
            except ValueError:
                row["wer"] = 0.0
            rows.append(row)

    os.makedirs(os.path.dirname(top_failures_csv) or ".", exist_ok=True)

    # 1) Top failures by CER.
    successful = [r for r in rows if r["success"] == 1]
    worst = sorted(successful, key=lambda r: (r["cer"], r["wer"]), reverse=True)[:top_k]
    with open(top_failures_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "engine",
            "sample_index",
            "image_path",
            "cer",
            "wer",
            "ground_truth",
            "prediction",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in worst:
            writer.writerow({k: r.get(k, "") for k in fields})

    # 2) Character confusion counts.
    confusions = Counter()
    for r in successful:
        gt = str(r.get("ground_truth", ""))
        pred = str(r.get("prediction", ""))
        for g, p in _align_chars(gt, pred):
            if g != p:
                confusions[(g, p)] += 1

    with open(confusion_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ground_truth_char", "predicted_char", "count"])
        writer.writeheader()
        for (g, p), count in confusions.most_common(300):
            writer.writerow({"ground_truth_char": g, "predicted_char": p, "count": count})

    # 3) Engine-level summary.
    agg = defaultdict(lambda: {"total": 0, "failed": 0, "cer_sum": 0.0, "wer_sum": 0.0})
    for r in rows:
        name = r.get("engine", "unknown")
        agg[name]["total"] += 1
        if r["success"] == 0:
            agg[name]["failed"] += 1
            continue
        agg[name]["cer_sum"] += r["cer"]
        agg[name]["wer_sum"] += r["wer"]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "engine",
                "total_rows",
                "failed_rows",
                "failure_rate",
                "mean_cer_success_only",
                "mean_wer_success_only",
            ],
        )
        writer.writeheader()
        for engine, stats in sorted(agg.items()):
            success_rows = max(stats["total"] - stats["failed"], 0)
            mean_cer = stats["cer_sum"] / success_rows if success_rows else 0.0
            mean_wer = stats["wer_sum"] / success_rows if success_rows else 0.0
            writer.writerow(
                {
                    "engine": engine,
                    "total_rows": stats["total"],
                    "failed_rows": stats["failed"],
                    "failure_rate": stats["failed"] / max(stats["total"], 1),
                    "mean_cer_success_only": mean_cer,
                    "mean_wer_success_only": mean_wer,
                }
            )

    print(f"Saved top failures to {top_failures_csv}")
    print(f"Saved confusion matrix rows to {confusion_csv}")
    print(f"Saved engine summary to {summary_csv}")


if __name__ == "__main__":
    run_error_analysis()
