import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


@dataclass
class EngineCostModel:
    """Cost model for each OCR engine."""
    tesseract_cost: float = 0.0
    custom_crnn_cost: float = 0.001
    trocr_cost: float = 0.05
    paddleocr_cost: float = 0.02
    google_vision_cost: float = 0.015
    cloud_api_cost: float = 0.10
    classifier_cost: float = 0.0001
    preprocessing_cost: float = 0.0005


class CostAnalyzer:
    def __init__(self, cost_model: EngineCostModel = None):
        self.cost_model = cost_model or EngineCostModel()
        if self.cost_model.cloud_api_cost <= 0:
            self.cost_model.cloud_api_cost = self.cost_model.google_vision_cost

    def analyze_routing_cost(
        self,
        routing_stats: dict,
        total_pages: int,
    ) -> dict:
        """
        Calculate total and per-page cost for intelligent routing
        vs alternative approaches.
        """
        cm = self.cost_model
        easy = routing_stats.get("easy", 0)
        medium = routing_stats.get("medium", 0)
        hard = routing_stats.get("hard", 0)

        overhead = total_pages * (cm.classifier_cost + cm.preprocessing_cost)

        routing_cost = (
            easy * cm.tesseract_cost +
            medium * cm.custom_crnn_cost +
            hard * cm.trocr_cost +
            overhead
        )

        all_cloud = total_pages * cm.cloud_api_cost
        all_trocr = total_pages * cm.trocr_cost + overhead
        all_tesseract = total_pages * cm.tesseract_cost + overhead

        return {
            "routing_cost": {
                "total": routing_cost,
                "per_page": routing_cost / max(total_pages, 1),
                "breakdown": {
                    "easy": easy * cm.tesseract_cost,
                    "medium": medium * cm.custom_crnn_cost,
                    "hard": hard * cm.trocr_cost,
                    "overhead": overhead,
                },
            },
            "all_cloud_cost": {"total": all_cloud, "per_page": cm.cloud_api_cost},
            "all_trocr_cost": {"total": all_trocr, "per_page": all_trocr / max(total_pages, 1)},
            "all_tesseract_cost": {"total": all_tesseract, "per_page": all_tesseract / max(total_pages, 1)},
            "savings_vs_cloud": (1 - routing_cost / max(all_cloud, 1e-10)) * 100,
            "savings_vs_trocr": (1 - routing_cost / max(all_trocr, 1e-10)) * 100,
        }

    def generate_cost_chart(self, analysis: dict, output_path: str = "reports/cost_comparison.png") -> None:
        """Generate bar chart comparing costs."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        approaches = ["All Tesseract", "All TrOCR", "All Cloud API", "Intelligent Routing"]
        costs = [
            analysis["all_tesseract_cost"]["total"],
            analysis["all_trocr_cost"]["total"],
            analysis["all_cloud_cost"]["total"],
            analysis["routing_cost"]["total"],
        ]

        colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(approaches, costs, color=colors)

        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Cost Comparison: OCR Approaches")

        for bar, cost in zip(bars, costs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"${cost:.2f}", ha="center", va="bottom", fontweight="bold",
            )

        savings = analysis["savings_vs_cloud"]
        ax.annotate(
            f"Savings vs Cloud: {savings:.0f}%",
            xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", fontsize=12, fontweight="bold", color="#2196F3",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Cost chart saved to {output_path}")
