"""Load and clean data for the student score regression example."""
from pathlib import Path
import pandas as pd

FEATURES = [
    "study_hours_per_day", "attendance_percentage", "previous_exam_score",
    "sleep_hours", "assignment_completion_rate", "extracurricular_participation",
]
TARGET = "academic_score"
CATEGORICAL_FEATURES = ["extracurricular_participation"]


def load_clean_data(path: str | Path) -> pd.DataFrame:
    """Read CSV, remove exact duplicate rows, validate columns and target."""
    df = pd.read_csv(path).drop_duplicates().copy()
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df[required]
    # Rows without a target cannot be used for supervised learning.
    return df.dropna(subset=[TARGET]).reset_index(drop=True)
