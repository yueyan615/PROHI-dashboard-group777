import streamlit as st

st.set_page_config(
    page_title="Welcome 👋",
    page_icon="./img/logo1.png",
    # layout="wide"
)


############################ SIDEBAR
### Logo
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  

img_body = st.sidebar.image("./img/logo_6.svg", use_container_width = False )
st.sidebar.caption("© 2025 Group 777 | Project Management and Tools for Health Informatics (PROHI)")


############################ MAIN BODY

"""
# Welcome to ObesityVision
"""
# Intro (one-liner)
st.markdown(
    "A concise, teaching-focused **web dashboard** for exploring obesity-related data, "
    "understanding drivers, predicting outcomes with a pre-trained model, and explaining model decisions."
)

st.divider()

# Four purposes — concise and outcome-oriented
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Descriptive")
    st.markdown(
        "Understand the **baseline profile** of the dataset — who/what/how much — and surface key **patterns** at a glance."
    )

with col2:
    st.markdown("### 🩺 Diagnostic")
    st.markdown(
        "Reveal **relationships and likely drivers** behind differences across groups to inform **why** outcomes vary."
    )

with col1:
    st.markdown("### 🎯 Predictive")
    st.markdown(
        "Generate **obesity-level estimates** for new cases using a **pre-trained model** (loaded from file) with confidence scores."
    )

with col2:
    st.markdown("### 🧩 Prescriptive (SHAP)")
    st.markdown(
        "Explain **which features drive each prediction** and in what direction, supporting **actionable interpretation**."
    )

st.divider()

# Compact data + disclaimer
st.markdown(
    "**Data note:** ~2,100 rows, ~17 features; class imbalance addressed via oversampling (e.g., SMOTE).  \n"
    "**Disclaimer:** Educational use only — not medical advice."
)