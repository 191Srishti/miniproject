
import csv
from pathlib import Path
from datetime import datetime


class MonitoringLogger:

    def __init__(self):

        # Get the main project directory
        project_folder = Path(__file__).resolve().parent.parent

        # Create data folder automatically
        data_folder = project_folder / "data"
        data_folder.mkdir(parents=True, exist_ok=True)

        self.file_path = data_folder / "monitoring_data.csv"

        # Open or create CSV
        self.file = open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8"
        )

        self.columns = [
            "timestamp",
            "red",
            "green",
            "blue",
            "estimated_bpm",
            "stable_bpm",
            "peak_ratio",
            "sampling_rate",
            "signal_duration",
            "signal_status"
        ]

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.columns
        )

        if self.file.tell() == 0:
            self.writer.writeheader()
            self.file.flush()

        print("CSV LOGGER INITIALIZED!")
        print("CSV FILE LOCATION:", self.file_path)

    def log_reading(
        self,
        red=None,
        green=None,
        blue=None,
        estimated_bpm=None,
        stable_bpm=None,
        peak_ratio=None,
        sampling_rate=None,
        signal_duration=None,
        signal_status="Unavailable"
    ):

        self.writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "red": red,
            "green": green,
            "blue": blue,
            "estimated_bpm": estimated_bpm,
            "stable_bpm": stable_bpm,
            "peak_ratio": peak_ratio,
            "sampling_rate": sampling_rate,
            "signal_duration": signal_duration,
            "signal_status": signal_status
        })

        self.file.flush()

    def close(self):
        self.file.close()
        print("CSV LOGGER CLOSED")

