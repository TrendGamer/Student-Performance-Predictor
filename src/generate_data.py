# Student Performance Predictor
"""Create a deterministic, documented synthetic student dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "student_data.csv"


def make_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate plausible demonstration data; this is not real student evidence."""
    rng = np.random.default_rng(seed)
    study = rng.uniform(0.25, 8.0, n)
    attendance = rng.uniform(55, 100, n)
    previous = rng.uniform(25, 98, n)
    sleep = np.clip(rng.normal(7.2, 1.15, n), 4, 10)
    assignments = rng.uniform(35, 100, n)
    extracurricular = rng.choice(["No", "Yes"], n, p=[0.58, 0.42])
    # A transparent synthetic rule plus noise. Scores are bounded to 0..100.
    noise = rng.normal(0, 7, n)
    score = (18 + 3.1 * study + 0.20 * (attendance - 55)
             + 0.34 * previous + 0.12 * (assignments - 35)
             + 1.25 * (sleep - 7) + 1.5 * (extracurricular == "Yes") + noise)
    df = pd.DataFrame({
        "study_hours_per_day": study,
        "attendance_percentage": attendance,
        "previous_exam_score": previous,
        "sleep_hours": sleep,
        "assignment_completion_rate": assignments,
        "extracurricular_participation": extracurricular,
        "academic_score": np.clip(score, 0, 100),
    }).round(2)
    # Add a small number of missing feature values and duplicate records to
    # demonstrate the preprocessing steps in a controlled, reproducible way.
    for col in ["study_hours_per_day", "attendance_percentage", "sleep_hours"]:
        positions = rng.choice(n, size=max(1, n // 100), replace=False)
        df.loc[positions, col] = np.nan
    return pd.concat([df, df.iloc[:8]], ignore_index=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make_dataset().to_csv(OUT, index=False)
    print(f"Wrote {len(make_dataset())} reproducible synthetic rows to {OUT}")
