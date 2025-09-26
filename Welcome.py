import streamlit as st
import datetime as dt

st.set_page_config(
    page_title="Welcome 👋",
    page_icon="./img/logo1.png",
    layout="wide"
)

############################ SIDEBAR
# Brand
# If your Streamlit version supports st.logo (>=1.31). Otherwise, remove it.
st.logo("./img/logo.svg", size="large")

st.sidebar.image("./img/logo_6.svg", use_container_width=False)
st.sidebar.divider()
year = dt.date.today().year
st.sidebar.caption(f"© {year} Group 777 · Project Management and Tools for Health Informatics (PROHI)")

############################ MAIN BODY
st.title("Welcome to ObesityVision")

# Intro (one-liner)
st.markdown(
    "An interactive web dashboard to explore obesity-related data, understand key drivers, "
    "predict outcomes with a pre-trained model, and explain model decisions."
)

st.divider()

# Lead-in
st.markdown("## What you can do here")
st.markdown(
    "Four outcome-oriented modules that take you from exploring the data to explaining decisions—clearly and responsibly."
)

# Row 1
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### 📊 Descriptive")
    st.markdown(
        "Understand the baseline profile of the dataset — who/what/how much — and surface key patterns at a glance."
    )
with c2:
    st.markdown("### 🩺 Diagnostic")
    st.markdown(
        "Reveal relationships and likely drivers behind differences across groups to inform why outcomes vary."
    )

# Row 2
c3, c4 = st.columns(2, gap="large")
with c3:
    st.markdown("### 🎯 Predictive")
    st.markdown(
        "Generate obesity-level estimates for new cases using a pre-trained model (loaded from file) with confidence scores."
    )
with c4:
    st.markdown("### 🧭 Prescriptive")
    st.markdown(
        "Use SHAP to show which features drive each prediction and in what direction—for actionable interpretation."
    )

st.divider()

# Footer in main body (keep minimal here; full footer sits in sidebar)
st.caption("See About for more data details.")
