"""Student Performance Predictor: a transparent Streamlit demo."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_performance_model.pkl"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="wide")
st.title("📚 Student Performance Predictor")
st.caption("A beginner-friendly machine learning demonstration")
st.info("This demo uses synthetic data. A prediction is an estimate for learning purposes, not a definitive judgment about any student.")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Follow the setup steps in README.md to generate the dataset and train it.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
st.markdown("Enter a few learning and wellbeing factors to see the model's estimated score.")
left, right = st.columns(2)
with left:
    study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.25)
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    previous = st.slider("Previous exam score", 0, 100, 65)
with right:
    sleep = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.25)
    assignments = st.slider("Assignment completion (%)", 0, 100, 80)
    extracurricular = st.selectbox("Extracurricular participation", ["No", "Yes"])

if st.button("Predict Performance", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "study_hours_per_day": study,
        "attendance_percentage": attendance,
        "previous_exam_score": previous,
        "sleep_hours": sleep,
        "assignment_completion_rate": assignments,
        "extracurricular_participation": extracurricular,
    }])
    score = float(model.predict(row)[0])
    score = max(0.0, min(100.0, score))
    category = "Excellent" if score >= 85 else "Good" if score >= 70 else "Average" if score >= 50 else "Needs Improvement"
    st.subheader("Estimated result")
    col1, col2 = st.columns([1, 2])
    col1.metric("Estimated academic score", f"{score:.1f} / 100", category)
    col2.progress(int(round(score)), text=f"{category} (illustrative bands)")
    st.caption("Category bands are simple demo labels: Excellent ≥85, Good ≥70, Average ≥50, otherwise Needs Improvement.")
    st.subheader("What the model considers")
    st.write("The estimate is based on all six inputs. In this synthetic dataset, the score was generated from a documented rule with random noise. These patterns are educational associations, not proof that changing any one factor causes a score change.")
    if METRICS_PATH.exists():
        import json
        results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(f"Selected model: {results['selected_model']} · Holdout MAE: {results['models'][results['selected_model']]['MAE']:.2f} points")

with st.expander("About this project"):
    st.write("The project compares Linear Regression, which fits a simple weighted relationship, with Random Forest, which averages many decision trees. See README.md and the EDA notebook for the full workflow.")
