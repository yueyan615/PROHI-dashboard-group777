import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
    # layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_cleaned.parquet'
df = pd.read_parquet(file_name)

############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  

# st.sidebar.markdown("# Prescriptive Analytics 🧭")


############################ MAIN BODY
""" # Prescriptive Analytics"""

"""
This section provides explanations for the model’s predictions using SHAP. Users can see which factors contribute most to obesity classification, both for individual cases and overall, and gain actionable insights.
"""
st.markdown("<br>", unsafe_allow_html=True)



########################### 1
"""## Obesity Dataset"""
