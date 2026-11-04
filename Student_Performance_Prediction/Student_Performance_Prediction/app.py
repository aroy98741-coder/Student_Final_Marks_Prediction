"""Streamlit app.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import load_model, predict_marks  # noqa: E402

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")
st.title("🎓 Student Final Marks Predictor")
st.write("Enter a student's details to predict the final exam marks (out of 100).")

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

attendance = st.slider("Attendance (%)", 50, 100, 85)
test1 = st.slider("Internal Test 1 (out of 40)", 0, 40, 32)
test2 = st.slider("Internal Test 2 (out of 40)", 0, 40, 32)
assignment = st.slider("Assignment score (out of 10)", 0, 10, 7)
hours = st.slider("Daily study hours", 0, 8, 3)

if st.button("Predict"):
    score = predict_marks(attendance, test1, test2, assignment, hours, model=model)
    st.metric("Predicted final marks", f"{score:.1f} / 100")
    st.progress(min(int(score), 100))
    st.caption("Estimate from a model trained on about 2,000 students; typical error is about 3-4 marks.")
