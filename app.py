import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("### Predict final student marks using AI")

# Sidebar
st.sidebar.header("📊 Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
study_hours = st.sidebar.slider("Study Hours per Day", 0.0, 12.0, 3.0)
internal_marks = st.sidebar.slider("Internal Marks", 0, 30, 15)

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Student Inputs")

    st.metric("Attendance", f"{attendance}%")
    st.metric("Study Hours", f"{study_hours} hrs/day")
    st.metric("Internal Marks", internal_marks)

with col2:
    st.subheader("📈 Performance Insights")

    chart_data = pd.DataFrame({
        "Factors": ["Attendance", "Study Hours", "Internal Marks"],
        "Values": [attendance, study_hours*10, internal_marks]
    })

    st.bar_chart(chart_data.set_index("Factors"))

# Prediction button
if st.button("🔍 Predict Final Marks"):

    prediction = model.predict([[attendance, study_hours, internal_marks]])

    st.success(f"🎯 Predicted Final Marks: {prediction[0]:.2f}")

    # Progress visualization
    progress = int(prediction[0])

    st.subheader("📊 Expected Performance Level")

    st.progress(min(progress, 100))

    if prediction[0] > 80:
        st.info("🏆 Excellent Performance")
    elif prediction[0] > 60:
        st.info("👍 Good Performance")
    elif prediction[0] > 40:
        st.warning("⚠️ Average Performance")
    else:
        st.error("❗ Needs Improvement")
