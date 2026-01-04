#!/usr/bin/env python3
"""
All-in-one Prophet training script on Peyton Manning Wikipedia pageviews.

What it does:
- Loads CSV from GitHub (facebook/prophet examples)
- Prepares data to Prophet's (ds, y) schema
- Trains a Prophet model
- Forecasts N future periods
- Saves model (joblib), forecast CSV, and plots (PNG)
- Demonstrates loading the saved model and predicting again
- Optionally runs Prophet's cross-validation to report metrics

Usage:
    pip install prophet pandas matplotlib joblib
    python prophet_train_peyton.py --periods 365 --freq D --cv
"""

# import argparse
# from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from joblib import dump, load
import matplotlib.pyplot as plt


DATA_URL = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Ensure correct schema for Prophet
    # if "ds" not in df.columns or "y" not in df.columns:
    #     # The file already has ds/y, but this is a guard if it ever changes
    #     rename_map = {}
    #     if "date" in df.columns:
    #         rename_map["date"] = "ds"
    #     if "value" in df.columns:
    #         rename_map["value"] = "y"
    #     df = df.rename(columns=rename_map)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna(subset=["y"])
    return df


def train_prophet(df: pd.DataFrame) -> Prophet:
    # A sensible baseline model; tweak if needed
    m = Prophet(
        # yearly_seasonality="auto",
        # weekly_seasonality="auto",
        # daily_seasonality=False,
        # seasonality_mode="additive",
        # changepoint_prior_scale=0.05,  # modest flexibility in trend
    )
    m.fit(df)
    return m


# def make_and_save_plots(model: Prophet, forecast: pd.DataFrame, outdir: Path):
def make_and_save_plots(model: Prophet, forecast: pd.DataFrame):
    # outdir.mkdir(parents=True, exist_ok=True)

    # Forecast plot
    fig1 = model.plot(forecast)
    fig1.tight_layout()
    # fig_path = outdir / "forecast.png"
    fig_path = "forecast.png"
    fig1.savefig(fig_path, dpi=150)
    plt.close(fig1)

    # Components plot
    fig2 = model.plot_components(forecast)
    fig2.tight_layout()
    # comp_path = outdir / "components.png"
    comp_path = "components.png"
    fig2.savefig(comp_path, dpi=150)
    plt.close(fig2)

    print(f"[saved] {fig_path}")
    print(f"[saved] {comp_path}")


def main():
    # parser = argparse.ArgumentParser(description="Train & save a Prophet model on the Peyton Manning pageviews dataset.")
    # parser.add_argument("--periods", type=int, default=365, help="Forecast horizon (# of future periods). Default: 365")
    # parser.add_argument("--freq", type=str, default="D", help="Pandas frequency for future periods (e.g., D, W, MS). Default: D")
    # parser.add_argument("--outdir", type=str, default="artifacts", help="Output directory for artifacts. Default: artifacts")
    # parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots.")
    # parser.add_argument("--cv", action="store_true", help="Run cross-validation and print performance metrics.")
    # args = parser.parse_args()

    # outdir = Path(args.outdir)
    # outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print("[info] loading data…")
    df = load_data(DATA_URL)
    print(f"[info] loaded {len(df):,} rows from {DATA_URL}")

    # 2) Train model
    print("[info] training Prophet model…")
    model = train_prophet(df)

    # 3) Forecast
    # print(f"[info] making future dataframe: periods={args.periods}, freq={args.freq}")
    # future = model.make_future_dataframe(periods=args.periods, freq=args.freq)
    future = model.make_future_dataframe(periods=365, freq='D')  # 30 days ahead
    forecast = model.predict(future)

    # 4) Save forecast
    # forecast_csv = outdir / "forecast.csv"
    forecast_csv = "forecast.csv"
    forecast.loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_csv, index=False)
    print(f"[saved] {forecast_csv}")

    # 5) Save model
    # model_path = outdir / "prophet_model.joblib"
    model_path = "prophet_model.joblib"
    dump(model, model_path)
    print(f"[saved] {model_path}")

    # 6) Optional: plots
    # if not args.no_plots:
    #     print("[info] generating plots…")
    #     make_and_save_plots(model, forecast, outdir)
    make_and_save_plots(model, forecast)

    # 7) Demo: load model and predict again (short horizon)
    print("[info] reloading model and predicting 30 more periods…")
    loaded = load(model_path)
    # future2 = loaded.make_future_dataframe(periods=30, freq=args.freq)
    future2 = loaded.make_future_dataframe(periods=365, freq='D')
    forecast2 = loaded.predict(future2)
    # quick_csv = outdir / "forecast_loaded_model.csv"
    quick_csv = "forecast_loaded_model.csv"
    forecast2.loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(quick_csv, index=False)
    print(f"[saved] {quick_csv}")

    # # 8) Optional: Cross-validation (rolling-origin evaluation)
    # if args.cv:
    #     print("[info] running cross-validation… (this can take a bit)")
    #     # For this daily series (~2008–2016), these are reasonable demo settings:
    #     df_cv = cross_validation(
    #         model,
    #         initial="1095 days",   # 3 years of initial training
    #         horizon="180 days",    # 6 months horizon
    #         period="90 days",      # step between cutoffs
    #     )
    #     df_p = performance_metrics(df_cv)
    #     metrics_csv = outdir / "cv_metrics.csv"
    #     df_p.to_csv(metrics_csv, index=False)
    #     print(f"[saved] {metrics_csv}")
    #     print(df_p.head())


if __name__ == "__main__":
    main()
