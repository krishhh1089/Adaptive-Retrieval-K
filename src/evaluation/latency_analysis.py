"""
src/evaluation/latency_analysis.py

Comprehensive latency profiling, bottleneck identification, and visualization generator
for Load-Aware SAGE baseline evaluations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Set headless backend for matplotlib
os.environ["MPLCONFIGDIR"] = str(Path.home() / ".cache" / "matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger

logger = setup_logger("latency_analyzer")


class LatencyAnalyzer:
    """
    Performs stage-wise latency decomposition, bottleneck analysis,
    and generates research-quality comparison charts.
    """

    def __init__(self, query_results_df: pd.DataFrame, output_dir: Path):
        self.df = query_results_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stage_columns = [
            "dense_retrieval_ms",
            "sparse_retrieval_ms",
            "hybrid_fusion_ms",
            "feature_extraction_ms",
            "policy_inference_ms",
            "context_preparation_ms",
            "llm_generation_ms",
        ]
        self.stage_display_names = {
            "dense_retrieval_ms": "Dense Retrieval (FAISS)",
            "sparse_retrieval_ms": "Sparse Retrieval (BM25)",
            "hybrid_fusion_ms": "RRF Hybrid Fusion",
            "feature_extraction_ms": "Probe Feature Extraction",
            "policy_inference_ms": "Policy Prediction",
            "context_preparation_ms": "Context Preparation",
            "llm_generation_ms": "LLM Generation (Groq)",
        }

    def compute_stage_summary(self) -> pd.DataFrame:
        """
        Calculates mean, P50, P95, and percentage of total latency for each stage per configuration.
        """
        rows = []
        configs = self.df["config_name"].unique()

        for cfg in configs:
            sub = self.df[self.df["config_name"] == cfg]
            total_mean = sub["total_end_to_end_ms"].mean()

            for stage in self.stage_columns:
                if stage in sub.columns:
                    vals = sub[stage].dropna().values
                    mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
                    p50_val = float(np.percentile(vals, 50)) if len(vals) > 0 else 0.0
                    p95_val = float(np.percentile(vals, 95)) if len(vals) > 0 else 0.0
                    pct = (mean_val / total_mean * 100.0) if total_mean > 0 else 0.0
                else:
                    mean_val, p50_val, p95_val, pct = 0.0, 0.0, 0.0, 0.0

                rows.append({
                    "config_name": cfg,
                    "stage_key": stage,
                    "stage_name": self.stage_display_names.get(stage, stage),
                    "mean_latency_ms": round(mean_val, 2),
                    "p50_latency_ms": round(p50_val, 2),
                    "p95_latency_ms": round(p95_val, 2),
                    "pct_of_total": round(pct, 2),
                })

        summary_df = pd.DataFrame(rows)
        csv_path = self.output_dir / "stage_latency.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved stage latency breakdown to: {csv_path}")
        return summary_df

    def identify_bottlenecks(self) -> Dict[str, Dict[str, Any]]:
        """
        Identifies the primary and secondary latency bottlenecks for each configuration.
        """
        bottlenecks = {}
        configs = self.df["config_name"].unique()

        for cfg in configs:
            sub = self.df[self.df["config_name"] == cfg]
            stage_means = {}
            for stage in self.stage_columns:
                if stage in sub.columns:
                    stage_means[self.stage_display_names.get(stage, stage)] = sub[stage].mean()

            sorted_stages = sorted(stage_means.items(), key=lambda x: x[1], reverse=True)
            total_mean = sub["total_end_to_end_ms"].mean()

            primary_stage, primary_ms = sorted_stages[0]
            secondary_stage, secondary_ms = sorted_stages[1] if len(sorted_stages) > 1 else ("N/A", 0.0)

            bottlenecks[cfg] = {
                "total_mean_ms": round(total_mean, 2),
                "primary_bottleneck": primary_stage,
                "primary_latency_ms": round(primary_ms, 2),
                "primary_share_pct": round(primary_ms / total_mean * 100.0, 2) if total_mean > 0 else 0.0,
                "secondary_bottleneck": secondary_stage,
                "secondary_latency_ms": round(secondary_ms, 2),
                "secondary_share_pct": round(secondary_ms / total_mean * 100.0, 2) if total_mean > 0 else 0.0,
            }

        return bottlenecks

    def generate_all_visualizations(self) -> List[Path]:
        """
        Renders and saves all 6 research charts to the output directory.
        """
        generated_plots = []
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Average End-to-End Latency Comparison
        p1 = self._plot_bar_metric(
            metric_col="total_end_to_end_ms",
            title="Average End-to-End Latency by Retrieval Configuration",
            ylabel="Mean Latency (ms)",
            filename="baseline_comparison.png",
            color="#2b5c8f",
        )
        generated_plots.append(p1)

        # 2. P95 Latency Comparison (SLO compliance)
        p2 = self._plot_p95_comparison()
        generated_plots.append(p2)

        # 3. Stage-wise Latency Breakdown (Stacked Bar)
        p3 = self._plot_stage_breakdown()
        generated_plots.append(p3)

        # 4. Average K Comparison
        p4 = self._plot_bar_metric(
            metric_col="selected_k",
            title="Average Retrieval Budget (K) Selected",
            ylabel="Average K",
            filename="average_k_comparison.png",
            color="#3a86ff",
        )
        generated_plots.append(p4)

        # 5. Token Usage Comparison
        p5 = self._plot_token_comparison()
        generated_plots.append(p5)

        # 6. Quality vs. Latency Trade-off
        p6 = self._plot_quality_vs_latency()
        generated_plots.append(p6)

        return generated_plots

    def _plot_bar_metric(self, metric_col: str, title: str, ylabel: str, filename: str, color: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = self.df.groupby("config_name")[metric_col].mean()
        # Sort in logical order: fixed_k2, fixed_k5, fixed_k10, fixed_k30, adaptive_sage
        order = [c for c in ["fixed_k2", "fixed_k5", "fixed_k10", "fixed_k30", "adaptive_sage"] if c in grouped.index]
        grouped = grouped.loc[order]

        bars = ax.bar(grouped.index, grouped.values, color=color, width=0.55, edgecolor="black", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Configuration", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def _plot_p95_comparison(self) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = self.df.groupby("config_name")["total_end_to_end_ms"].apply(lambda s: np.percentile(s, 95))
        order = [c for c in ["fixed_k2", "fixed_k5", "fixed_k10", "fixed_k30", "adaptive_sage"] if c in grouped.index]
        grouped = grouped.loc[order]

        bars = ax.bar(grouped.index, grouped.values, color="#e63946", width=0.55, edgecolor="black", alpha=0.85)
        ax.axhline(5000, color="darkred", linestyle="--", linewidth=1.5, label="Target SLO (5.0s)")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f} ms",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("P95 Request Latency (Tail Latency vs. SLO Target)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("P95 Latency (ms)", fontsize=11)
        ax.set_xlabel("Configuration", fontsize=11)
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = self.output_dir / "p95_comparison.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def _plot_stage_breakdown(self) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = [c for c in ["fixed_k2", "fixed_k5", "fixed_k10", "fixed_k30", "adaptive_sage"] if c in self.df["config_name"].unique()]

        data = {s: [] for s in self.stage_columns}
        for cfg in order:
            sub = self.df[self.df["config_name"] == cfg]
            for s in self.stage_columns:
                val = sub[s].mean() if s in sub.columns else 0.0
                data[s].append(val)

        bottom = np.zeros(len(order))
        colors = ["#457b9d", "#1d3557", "#2a9d8f", "#e76f51", "#f4a261", "#a8dadc", "#e63946"]

        for (stage, vals), color in zip(data.items(), colors):
            vals_arr = np.array(vals)
            ax.bar(
                order,
                vals_arr,
                bottom=bottom,
                label=self.stage_display_names.get(stage, stage),
                color=color,
                width=0.55,
                edgecolor="white",
            )
            bottom += vals_arr

        ax.set_title("Decomposed Latency Breakdown by Pipeline Stage", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Mean Stage Latency (ms)", fontsize=11)
        ax.set_xlabel("Configuration", fontsize=11)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = self.output_dir / "latency_breakdown.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def _plot_token_comparison(self) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = [c for c in ["fixed_k2", "fixed_k5", "fixed_k10", "fixed_k30", "adaptive_sage"] if c in self.df["config_name"].unique()]

        prompt_tokens = [self.df[self.df["config_name"] == c]["prompt_tokens"].mean() for c in order]
        completion_tokens = [self.df[self.df["config_name"] == c]["completion_tokens"].mean() for c in order]

        x = np.arange(len(order))
        width = 0.35

        ax.bar(x - width / 2, prompt_tokens, width, label="Prompt Tokens (Input Context)", color="#457b9d", edgecolor="black")
        ax.bar(x + width / 2, completion_tokens, width, label="Completion Tokens (Generated)", color="#f4a261", edgecolor="black")

        ax.set_title("Token Usage Breakdown Across Retrieval Configurations", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Average Token Count", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = self.output_dir / "token_comparison.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def _plot_quality_vs_latency(self) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        configs = [c for c in ["fixed_k2", "fixed_k5", "fixed_k10", "fixed_k30", "adaptive_sage"] if c in self.df["config_name"].unique()]

        for cfg in configs:
            sub = self.df[self.df["config_name"] == cfg]
            lat = sub["total_end_to_end_ms"].mean()
            # Quality metric: Token F1 or Retrieval Hit Rate
            quality = sub["token_f1"].mean() * 100.0 if "token_f1" in sub.columns else sub["retrieval_hit"].mean() * 100.0
            marker = "*" if cfg == "adaptive_sage" else "o"
            size = 180 if cfg == "adaptive_sage" else 110
            color = "#e63946" if cfg == "adaptive_sage" else "#1d3557"

            ax.scatter(lat, quality, s=size, marker=marker, color=color, zorder=5)
            ax.annotate(
                f" {cfg}\n (K={sub['selected_k'].mean():.1f})",
                xy=(lat, quality),
                xytext=(6, -6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold" if cfg == "adaptive_sage" else "normal",
            )

        ax.set_title("Quality vs. Latency Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean End-to-End Latency (ms)", fontsize=11)
        ax.set_ylabel("Answer Quality (Token F1 %)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = self.output_dir / "quality_vs_latency.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

