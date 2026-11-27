
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_DIR / "data" / "monitoring_data.csv"

OUTPUT_DIR = PROJECT_DIR / "analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load CSV data
# --------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("\nActual signal status values:")
print(df["signal_status"].value_counts(dropna=False))
print("\nBPM column information:")
print("Estimated BPM non-null:", df["estimated_bpm"].notna().sum())
print("Stable BPM non-null:", df["stable_bpm"].notna().sum())

print("\nEstimated BPM examples:")
print(df["estimated_bpm"].dropna().head(10).to_list())
print("\nFirst 10 records:")
print(df.head(10).to_string())

if df.empty:
    raise ValueError("The CSV file contains no monitoring records.")

required_columns = [
    "timestamp",
    "estimated_bpm",
    "signal_status"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")


# --------------------------------------------------
# Clean and prepare data
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["estimated_bpm"] = pd.to_numeric(
    df["estimated_bpm"],
    errors="coerce"
)

df["stable_bpm"] = pd.to_numeric(
    df["stable_bpm"],
    errors="coerce"
)

df["peak_ratio"] = pd.to_numeric(
    df["peak_ratio"],
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# Only count actual accepted BPM estimates.
valid_bpm = df[
    df["estimated_bpm"].notna()
    & df["estimated_bpm"].between(42, 180)
]

# A signal was processed if it was not unavailable.
available_signal = df[
    df["signal_status"].isin(["Processed", "BPM valid", "BPM rejected"])
]

total_readings = len(df)
valid_readings = len(valid_bpm)
available_readings = len(available_signal)
rejected_readings = total_readings - valid_readings

bpm_availability = (
    valid_readings / total_readings * 100
    if total_readings else 0
)

signal_availability = (
    available_readings / total_readings * 100
    if total_readings else 0
)


# --------------------------------------------------
# Calculate heart-rate statistics
# --------------------------------------------------

if valid_readings > 0:

    average_bpm = valid_bpm["estimated_bpm"].mean()
    minimum_bpm = valid_bpm["estimated_bpm"].min()
    maximum_bpm = valid_bpm["estimated_bpm"].max()

else:
    average_bpm = None
    minimum_bpm = None
    maximum_bpm = None


# --------------------------------------------------
# Print report
# --------------------------------------------------

print("\n====================================")
print(" CONTACTLESS HEALTH MONITOR REPORT")
print("====================================")

print(f"Total readings: {total_readings}")
print(f"Valid BPM readings: {valid_readings}")
print(f"Rejected BPM readings: {rejected_readings}")

print(f"BPM availability: {bpm_availability:.2f}%")
print(f"Signal availability: {signal_availability:.2f}%")

if average_bpm is not None:
    print(f"Average BPM: {average_bpm:.2f}")
    print(f"Minimum BPM: {minimum_bpm:.2f}")
    print(f"Maximum BPM: {maximum_bpm:.2f}")
else:
    print("No valid BPM readings available.")

print("====================================")


# --------------------------------------------------
# Graph 1: Heart-rate trend
# --------------------------------------------------

if valid_readings > 0:

    plt.figure(figsize=(12, 5))

    plt.plot(
        valid_bpm["timestamp"],
        valid_bpm["estimated_bpm"],
        marker="o",
        markersize=3,
        label="Estimated BPM"
    )

    plt.axhline(
        average_bpm,
        linestyle="--",
        label=f"Average: {average_bpm:.1f} BPM"
    )

    plt.title("Webcam-Based Heart Rate Trend")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "heart_rate_trend.png",
        dpi=150
    )

    plt.show()

else:
    print("Skipping heart-rate graph: no valid readings.")


# --------------------------------------------------
# Graph 2: Reading status distribution
# --------------------------------------------------

status_counts = pd.Series({
    "Valid BPM": valid_readings,
    "Rejected BPM": rejected_readings
})

plt.figure(figsize=(8, 5))

plt.bar(
    status_counts.index,
    status_counts.values
)

plt.title("BPM Reading Status Distribution")
plt.ylabel("Number of Readings")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "reading_status.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Save experiment summary
# --------------------------------------------------

summary = pd.DataFrame([{
    "total_readings": total_readings,
    "valid_bpm_readings": valid_readings,
    "rejected_bpm_readings": rejected_readings,
    "bpm_availability_percent": round(bpm_availability, 2),
    "signal_availability_percent": round(signal_availability, 2),
    "average_bpm": round(average_bpm, 2)
        if average_bpm is not None else None,
    "minimum_bpm": round(minimum_bpm, 2)
        if minimum_bpm is not None else None,
    "maximum_bpm": round(maximum_bpm, 2)
        if maximum_bpm is not None else None
}])

summary.to_csv(
    OUTPUT_DIR / "experiment_summary.csv",
    index=False
)

print("\nAnalysis completed!")
print("Results saved in:", OUTPUT_DIR)
