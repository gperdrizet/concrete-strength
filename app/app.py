from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "gpr_models.pkl",
    PROJECT_ROOT / "models" / "gpr_model.pkl",
]
DATA_PATH = PROJECT_ROOT / "data" / "concrete_data.csv"
FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
]
MIX_FEATURES = FEATURES[:-1]
FEATURE_LABELS = {
    "cement": "Cement (kg/m^3)",
    "slag": "Blast Furnace Slag (kg/m^3)",
    "fly_ash": "Fly Ash (kg/m^3)",
    "water": "Water (kg/m^3)",
    "superplasticizer": "Superplasticizer (kg/m^3)",
    "coarse_agg": "Coarse Aggregate (kg/m^3)",
    "fine_agg": "Fine Aggregate (kg/m^3)",
}


def apply_theme():
    st.markdown(
        """
        <style>
            :root {
                --bg-0: #0a0f1f;
                --bg-1: #1b1230;
                --bg-2: #0d2e2a;
                --fg-main: #d8f3dc;
                --fg-muted: #b7dfc8;
                --purple-accent: #9b5de5;
                --green-accent: #2ec4b6;
                --card-bg: rgba(13, 23, 41, 0.78);
            }

            .stApp {
                background: radial-gradient(circle at 15% 20%, rgba(155, 93, 229, 0.26) 0%, transparent 35%),
                            radial-gradient(circle at 85% 80%, rgba(46, 196, 182, 0.24) 0%, transparent 42%),
                            linear-gradient(135deg, var(--bg-0), var(--bg-1) 48%, var(--bg-2));
                color: var(--fg-main);
            }

            .block-container {
                background: var(--card-bg);
                border: 1px solid rgba(155, 93, 229, 0.28);
                border-radius: 14px;
                padding: 1.5rem 1.5rem 2rem;
            }

            h1, h2, h3, label, p, div, span {
                color: var(--fg-main) !important;
            }

            .stCaption, .stMarkdown p {
                color: var(--fg-muted) !important;
            }

            .stMetric {
                border: 1px solid rgba(46, 196, 182, 0.25);
                border-radius: 10px;
                padding: 0.5rem;
                background: rgba(18, 34, 35, 0.45);
            }

            [data-baseweb="slider"] div[role="slider"] {
                background: linear-gradient(180deg, #6a4c93 0%, #2ec4b6 100%) !important;
                border: 2px solid rgba(216, 243, 220, 0.8) !important;
                box-shadow: 0 0 0 3px rgba(155, 93, 229, 0.22);
            }

            [data-baseweb="slider"] > div > div {
                background: linear-gradient(90deg, rgba(155, 93, 229, 0.8), rgba(46, 196, 182, 0.8)) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path.name
    candidate_list = "\n".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"No model file found. Looked for:\n{candidate_list}"
    )


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def make_slider(col, feature_name, min_value, max_value, default):
    return col.slider(
        FEATURE_LABELS[feature_name],
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(default),
        step=0.1,
    )


def build_time_curve_inputs(component_values, ages):
    rows = []
    for age in ages:
        row = {**component_values, "age": float(age)}
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def main():
    apply_theme()

    st.title("Concrete Compressive Strength Over Time")
    st.write(
        "Use the sliders to set a concrete mix. The chart shows predicted strength "
        "across curing age with a 95% confidence interval."
    )

    model, model_name = load_model()
    df = load_data()

    st.caption(f"Loaded model: {model_name}")

    bounds = df[MIX_FEATURES].agg(["min", "max", "median"])

    left, right = st.columns(2)
    component_values = {}

    for idx, feature in enumerate(MIX_FEATURES):
        target_col = left if idx % 2 == 0 else right
        component_values[feature] = make_slider(
            target_col,
            feature,
            bounds.loc["min", feature],
            bounds.loc["max", feature],
            bounds.loc["median", feature],
        )

    st.subheader("Predicted Strength Curve")
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    ages = np.arange(age_min, age_max + 1)
    X_curve = build_time_curve_inputs(component_values, ages)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X has feature names")
        mean_strength, std_strength = model.predict(X_curve, return_std=True)

    ci_half_width = 1.96 * std_strength
    lower = mean_strength - ci_half_width
    upper = mean_strength + ci_half_width

    current_age = st.slider(
        "Check prediction at a specific age (days)",
        min_value=age_min,
        max_value=age_max,
        value=28,
        step=1,
    )

    current_input = pd.DataFrame(
        [{**component_values, "age": float(current_age)}],
        columns=FEATURES,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X has feature names")
        current_mean, current_std = model.predict(current_input, return_std=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=upper,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(155, 93, 229, 0.24)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="95% Confidence Interval",
            hovertemplate=(
                "Age: %{x} days<br>"
                "Lower 95% CI: %{y:.2f} MPa"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=mean_strength,
            mode="lines",
            name="Predicted Mean",
            line=dict(color="#2ec4b6", width=3),
            hovertemplate=(
                "Age: %{x} days<br>"
                "Predicted strength: %{y:.2f} MPa"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_age],
            y=[current_mean[0]],
            mode="markers",
            name="Selected Day",
            marker=dict(color="#9b5de5", size=10, line=dict(color="#d8f3dc", width=1)),
            hovertemplate=(
                "Age: %{x} days<br>"
                "Predicted strength: %{y:.2f} MPa"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Predicted Concrete Strength vs. Age",
        xaxis_title="Age (days)",
        yaxis_title="Compressive Strength (MPa)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9, 16, 32, 0.86)",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(46, 196, 182, 0.30)", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(155, 93, 229, 0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(46, 196, 182, 0.18)", zeroline=False)

    st.plotly_chart(fig, use_container_width=True)

    current_ci = 1.96 * current_std[0]
    st.metric(
        label=f"Predicted Strength at Day {current_age}",
        value=f"{current_mean[0]:.2f} MPa",
        help=f"95% CI: {current_mean[0] - current_ci:.2f} to {current_mean[0] + current_ci:.2f} MPa",
    )


if __name__ == "__main__":
    main()
