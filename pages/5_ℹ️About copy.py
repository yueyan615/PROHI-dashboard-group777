# about.py
import streamlit as st
import datetime as dt

st.set_page_config(page_title="About ℹ️", page_icon="./img/logo1.png", layout="wide")

# Global brand (small horizontal logo across all pages)
st.logo("./img/logo.svg", size="large")

st.title("About")

# 1) Problem Description (non–data-science info)
st.markdown("""
## Problem Description
This project explores how lifestyle and physical-condition factors relate to obesity level categories and how a simple, explainable tool can help non-specialists interpret results. The goal is an interactive web dashboard that communicates insights clearly, supports classroom use, and offers responsible, non-clinical guidance.
""")

# 2) About the Dataset
st.markdown("""
## About the Dataset
This dashboard uses data from the UCI Machine Learning Repository.

Dataset name: *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*

Description:
The dataset contains information collected from individuals regarding their eating habits, physical condition, and lifestyle behaviors.
It was designed to predict obesity levels, classified into categories such as Underweight, Normal Weight, Overweight, and Obese based on WHO standards.
""")

# 3) Design Process (non–data-science info)
st.markdown("""
## Design Process
- Requirements: define four analytics paths (descriptive, diagnostic, predictive with a pre-trained model, prescriptive with SHAP explanations).
- Prototyping: low-fidelity navigation and copy; iterate wording for clarity and responsibility.
- Implementation: train models in Jupyter notebooks; deploy a Streamlit dashboard that loads the pre-trained model file for inference only.
- Validation: manual checks on usability/readability; include scope and disclaimer to avoid clinical interpretation.
- Deliverables: multi-tab dashboard (Welcome, Descriptive, Diagnostic, Predictive, Prescriptive, About) and project documentation.
""")

# 4) Dashboard Structure & Methods (concise)
st.markdown("""
## Dashboard Structure & Methods
- Descriptive: baseline profiles and patterns.
- Diagnostic: relationships and potential drivers of differences.
- Predictive: on-demand predictions using a pre-trained `.pkl` model (no training in the app).
- Prescriptive (explainability): SHAP-based explanations (single/batch) indicating which features drive predictions and in what direction.
""")

# 5) Team & Contact (explicit contact info required by the brief)
st.markdown("""
## About Our Team
This dashboard was created by Group 777 as part of the assignment for Project Management and Tools for Health Informatics (PROHI).

Team members:
- Weiqi Kong
- Yueyan Li
- Zsolt Fehér
- Christoffer Brändefors
- Naznin Akhtar

Contact:
- Email: group777@university.edu
- (Optional) Project repo: https://github.com/group777/obesityvision
""")

# 6) References (as provided)
st.markdown("""
## References
[1] Palechor FM, Manotas AH. Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data Brief. 2019;25:104344. DOI: 10.1016/j.dib.2019.104344.

[2] UCI Machine Learning Repository. Estimation of Obesity Levels Based on Eating Habits and Physical Condition [Internet]. 2019. Available from: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

[3] World Health Organization. Obesity and overweight. [Internet]. 2021. Available from: https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight

[4] Solomon DD, Khan S, Garg S, Gupta G, Almjally A, Alabduallah BI, et al. Hybrid Majority Voting: Prediction and Classification Model for Obesity. Diagnostics (Basel). 2023;13(15). DOI: 10.3390/diagnostics13152610.
""")

st.divider()
st.caption("Scope: educational and exploratory use; not medical advice.")

# Sidebar footer (recommended on all pages)
with st.sidebar:
    st.divider()
    st.caption(f"© {dt.date.today().year} Group 777 · Project Management and Tools for Health Informatics (PROHI)")
