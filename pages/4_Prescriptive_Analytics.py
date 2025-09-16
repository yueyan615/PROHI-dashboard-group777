import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
)


############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  

# st.sidebar.markdown("# Prescriptive Analytics 🧭")


############################ MAIN BODY
# load the dataset
file_name = './assets/ObesityDataSet_cleaned.csv'
df = pd.read_csv(file_name)


""" # Prescriptive Analytics 🧭"""