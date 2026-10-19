"""
Exploratory Data Analysis (EDA) Module for Subscription Revenue.

Generates and saves visual analysis figures to eda/figures/ including:
- Overall revenue trajectory
- Per-country revenue comparisons
- Revenue distributions across countries
- Monthly seasonality analysis
- Naive baseline comparison visualization
"""

import argparse
import logging
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_figures(input_path: str = "data/processed/revenue_timeseries.csv", output_dir: str = "eda/figures"):
    """Generate and save all exploratory data analysis figures."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # Figure 1: Overall Revenue Trend (Aggregate 'all')
    all_df = df[df["country"] == "all"].sort_values("date")
    if not all_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(all_df["date"], all_df["revenue"], marker="o", color="#2563eb", linewidth=2.5, label="Total Revenue")
        # Moving average trend
        all_df["ma_3"] = all_df["revenue"].rolling(window=3, min_periods=1).mean()
        plt.plot(all_df["date"], all_df["ma_3"], linestyle="--", color="#dc2626", linewidth=1.8, label="3-Period Moving Avg")
        plt.title("Total Subscription Revenue Over Time (All Countries Aggregate)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Revenue ($)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        trend_path = os.path.join(output_dir, "overall_revenue_trend.png")
        plt.savefig(trend_path, dpi=200)
        plt.close()
        logger.info("Saved %s", trend_path)

    # Figure 2: Country Revenue Comparison
    country_df = df[df["country"] != "all"].sort_values(["country", "date"])
    countries = country_df["country"].unique()
    if len(countries) > 0:
        plt.figure(figsize=(12, 7))
        colors = plt.cm.tab10.colors
        for idx, country in enumerate(countries):
            subset = country_df[country_df["country"] == country]
            plt.plot(subset["date"], subset["revenue"], marker="s", markersize=4,
                     linewidth=1.8, label=country, color=colors[idx % len(colors)])
        plt.title("Per-Country Subscription Revenue Trajectories", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Monthly Revenue ($)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()
        country_comp_path = os.path.join(output_dir, "country_revenue_comparison.png")
        plt.savefig(country_comp_path, dpi=200)
        plt.close()
        logger.info("Saved %s", country_comp_path)

    # Figure 3: Country Revenue Distribution (Total and Boxplot)
    if len(countries) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total revenue per country
        totals = country_df.groupby("country")["revenue"].sum().sort_values(ascending=False)
        axes[0].bar(totals.index, totals.values, color="#3b82f6", edgecolor="#1d4ed8")
        axes[0].set_title("Total Revenue by Country", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Total Revenue ($)", fontsize=11)
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)

        # Boxplot of monthly revenue
        box_data = [country_df[country_df["country"] == c]["revenue"].values for c in totals.index]
        axes[1].boxplot(box_data, tick_labels=totals.index, patch_artist=True,
                        boxprops=dict(facecolor="#93c5fd", color="#1e40af"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[1].set_title("Monthly Revenue Distribution by Country", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Monthly Revenue ($)", fontsize=11)
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        dist_path = os.path.join(output_dir, "country_revenue_distribution.png")
        plt.savefig(dist_path, dpi=200)
        plt.close()
        logger.info("Saved %s", dist_path)

    # Figure 4: Seasonality Analysis
    if not all_df.empty:
        all_df_season = all_df.copy()
        all_df_season["month"] = all_df_season["date"].dt.month_name()
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        monthly_avg = all_df_season.groupby("month")["revenue"].mean().reindex(month_order).dropna()
        
        plt.figure(figsize=(10, 5))
        plt.bar(monthly_avg.index, monthly_avg.values, color="#10b981", edgecolor="#047857")
        plt.title("Average Revenue by Calendar Month (Seasonality Analysis)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Month", fontsize=11)
        plt.ylabel("Average Revenue ($)", fontsize=11)
        plt.xticks(rotation=40)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        season_path = os.path.join(output_dir, "monthly_seasonality.png")
        plt.savefig(season_path, dpi=200)
        plt.close()
        logger.info("Saved %s", season_path)

    # Figure 5: Naive Baseline vs Actual Visualization
    if not all_df.empty:
        all_df_base = all_df.copy()
        all_df_base["naive_pred"] = all_df_base["revenue"].shift(1)
        valid_base = all_df_base.dropna(subset=["naive_pred"])

        plt.figure(figsize=(12, 6))
        plt.plot(valid_base["date"], valid_base["revenue"], marker="o", label="Actual Revenue", color="#1e293b", linewidth=2)
        plt.plot(valid_base["date"], valid_base["naive_pred"], linestyle="--", marker="x", label="Naive Baseline (t-1)", color="#f59e0b", linewidth=1.8)
        plt.title("Actual vs. Naive Last-Observation Baseline ('all' Series)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Revenue ($)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        base_path = os.path.join(output_dir, "baseline_visualization.png")
        plt.savefig(base_path, dpi=200)
        plt.close()
        logger.info("Saved %s", base_path)

    logger.info("EDA figure generation complete. All plots saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Exploratory Data Analysis (EDA) figures.")
    parser.add_argument("--input", type=str, default="data/processed/revenue_timeseries.csv", help="Input processed CSV file")
    parser.add_argument("--output-dir", type=str, default="eda/figures", help="Directory to save figures")
    args = parser.parse_args()

    generate_eda_figures(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
