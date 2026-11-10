
import pandas as pd

TARGET = "target_rtt_30s"

FEATURE_CANDIDATES = [
    "rtt_avg_ms",
    "rtt_avg_30s_ago",
    "rtt_avg_60s_ago",
    "jitter_ms",
    "packet_loss_pct",
    "latency_mean_5",
    "latency_max_5",
    "latency_std_5",
    "latency_change"
]

def prepare_features(df):

    required_columns = FEATURE_CANDIDATES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Maintain chronological order
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc").copy()

    # Same row filtering as notebook
    model_df = df[required_columns].dropna().copy()

    X = model_df[FEATURE_CANDIDATES].apply(
        pd.to_numeric, errors="raise"
    )

    y = pd.to_numeric(model_df[TARGET], errors="raise")

    print("Features used:", list(X.columns))
    print("Rows available:", len(X))

    return X, y
