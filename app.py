
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# -------------------- PATHS --------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "latency_model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "final.csv"

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Network Intelligence",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📡 Network Intelligence Dashboard")
st.caption(
    "ML-powered network latency forecasting | "
    "Prediction horizon: 30 seconds"
)

# -------------------- LOAD MODEL --------------------
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python -m src.train` first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

# -------------------- LOAD HISTORICAL DATA --------------------
history = None

if DATA_PATH.exists():
    try:
        history = pd.read_csv(DATA_PATH)
    except Exception:
        history = None

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.title("⚙️ Dashboard")
st.sidebar.write("Enter network measurements below.")
st.sidebar.caption("These are manual inputs, not live readings.")

defaults = {
    "rtt_avg_ms": 10.0,
    "jitter_ms": 2.0,
    "packet_loss_pct": 0.0,
    "gateway_rtt_avg_ms": 5.0,
    "tcp_connect_median_ms": 10.0,
}

with st.sidebar.form("network_inputs"):
    st.subheader("Current Measurements")

    values = {}

    for feature in features:
        maximum = 100.0 if feature == "packet_loss_pct" else None

        kwargs = {
            "label": feature.replace("_", " ").title(), 
            "min_value": 0.0,
            "value": float(defaults.get(feature, 0.0)),
            "step": 0.1,
            "key": f"input_{feature}",
        }

        if maximum is not None:
            kwargs["max_value"] = maximum

        values[feature] = st.number_input(**kwargs)

    submitted = st.form_submit_button(
        "🔮 Forecast Latency",
        use_container_width=True
    )

# -------------------- STORE PREDICTION --------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.input_values = None

if submitted:
    input_df = pd.DataFrame([values], columns=features)
    prediction = float(model.predict(input_df)[0])

    st.session_state.prediction = prediction
    st.session_state.input_values = values.copy()

# -------------------- NETWORK OVERVIEW --------------------
st.subheader("Network Overview")

if st.session_state.prediction is None:
    st.info(
        "Enter measurements in the sidebar and click "
        "'Forecast Latency' to get started." 
    )

else:
    prediction = st.session_state.prediction
    entered = st.session_state.input_values

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted RTT (+30 sec)",
        f"{prediction:.3f} ms"
    )

    col2.metric(
        "Current RTT",
        f"{entered.get('rtt_avg_ms', 0):.2f} ms"
    )

    col3.metric(
        "Jitter",
        f"{entered.get('jitter_ms', 0):.2f} ms"
    )

    col4.metric(
        "Packet Loss",
        f"{entered.get('packet_loss_pct', 0):.2f}%"
    )

    if prediction < 0:
        st.warning(
            "Negative latency is not physically meaningful. "
            "Review the model and its predictions."
        )
    elif prediction < 20:
        st.success(
            "Predicted RTT is below 20 ms. "
            "This is an illustrative threshold, not a universal health verdict."
        )
    else:
        st.warning(
            "Predicted RTT is 20 ms or higher. "
            "Interpret this using your network's normal operating range."
        )

    # -------------------- WHAT-IF SIMULATOR --------------------
    st.divider()
    st.subheader("🧪 What-if Scenario Simulator")

    st.write(
        "Change selected measurements to see how the model's "
        "prediction changes. This is a hypothetical scenario."
    )

    scenario = entered.copy()

    c1, c2, c3 = st.columns(3)

    if "rtt_avg_ms" in features:
        scenario["rtt_avg_ms"] = c1.number_input(
            "Scenario: Average RTT (ms)",
            min_value=0.0,
            value=float(entered["rtt_avg_ms"]),
            step=1.0,
            key="scenario_rtt"
        )

    if "jitter_ms" in features:
        scenario["jitter_ms"] = c2.number_input(
            "Scenario: Jitter (ms)",
            min_value=0.0,
            value=float(entered["jitter_ms"]),
            step=0.5,
            key="scenario_jitter"
        )

    if "packet_loss_pct" in features:
        scenario["packet_loss_pct"] = c3.number_input(
            "Scenario: Packet Loss (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(entered["packet_loss_pct"]),
            step=0.1,
            key="scenario_loss"
        )

    scenario_df = pd.DataFrame([scenario], columns=features)
    scenario_prediction = float(model.predict(scenario_df)[0])

    delta = scenario_prediction - prediction

    s1, s2 = st.columns(2)

    s1.metric(
        "Scenario Forecast",
        f"{scenario_prediction:.3f} ms"
    )

    s2.metric(
        "Change vs Original Forecast",
        f"{delta:+.3f} ms"
    )

    comparison = pd.DataFrame({
        "Scenario": ["Original Inputs", "What-if Inputs"],
        "Predicted RTT (ms)": [prediction, scenario_prediction]
    }).set_index("Scenario")

    st.bar_chart(comparison)

    # -------------------- DOWNLOAD REPORT --------------------
    st.divider()
    st.subheader("📥 Export Forecast")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_horizon_seconds": 30,
        **{f"input_{k}": v for k, v in entered.items()},
        "original_prediction_ms": prediction,
        **{f"scenario_{k}": v for k, v in scenario.items()},
        "scenario_prediction_ms": scenario_prediction,
    }

    report_df = pd.DataFrame([report])

    st.download_button(
        "Download Forecast Report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="network_forecast_report.csv",
        mime="text/csv"
    )

# -------------------- HISTORICAL LATENCY TREND --------------------
st.divider()
st.subheader("📈 Historical Network Latency")

required_columns = {"timestamp_utc", "rtt_avg_ms"}

if history is not None and required_columns.issubset(history.columns):

    chart_data = history[
        ["timestamp_utc", "rtt_avg_ms"]
    ].copy()

    chart_data["timestamp_utc"] = pd.to_datetime(
        chart_data["timestamp_utc"],
        errors="coerce"
    )

    chart_data["rtt_avg_ms"] = pd.to_numeric(
        chart_data["rtt_avg_ms"],
        errors="coerce"
    )

    chart_data = (
        chart_data.dropna()
        .sort_values("timestamp_utc")
        .tail(500)
    )

    if not chart_data.empty:
        chart_data = chart_data.set_index("timestamp_utc")

        st.line_chart(chart_data["rtt_avg_ms"])

        st.caption(
            "Showing up to 500 historical records. "
            "This chart does not represent live network traffic."
        )
    else:
        st.info("No valid historical latency measurements found.")

else:
    st.info(
        "Historical chart unavailable. Check that the CSV contains "
        "'timestamp_utc' and 'rtt_avg_ms'."
    )

# -------------------- FOOTER --------------------
st.divider()

st.caption(
    "Network Intelligence | ML-based latency forecasting. "
    "Predictions are estimates, not guarantees."
)
