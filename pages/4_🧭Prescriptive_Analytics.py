# you need "pip install streamlit-shap" if you haven't already installed it with the requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap


st.set_page_config(
    page_title="Prescriptive Analytics | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_cleaned.parquet'
df = pd.read_parquet(file_name)

############################ SIDEBAR
### Logo
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  

### Navigation on the page
st.sidebar.write("<br>", unsafe_allow_html=True)
def scroll_to(element_id: str):
    components.html(
        f"""
        <script>
        const t = window.parent.document.getElementById("{element_id}");
        if (t) t.scrollIntoView({{behavior: "smooth", block: "start"}});
        </script>
        """,
        height=0,
    )

with st.sidebar:
    sec = st.radio("Navigate on the page", ["Prescriptive Analytics","SHAP Summary Plot", "SHAP Force Plot"], index=0)
    mapping = {"Prescriptive Analytics": "Prescriptive Analytics", "SHAP Summary Plot": "SHAP Summary Plot", "SHAP Force Plot": "SHAP Force Plot"}
    scroll_to(mapping[sec])


############################ MAIN BODY
st.markdown('<div id="Prescriptive Analytics"></div>', unsafe_allow_html=True)
""" # Prescriptive Analytics"""

"""
This section provides explanations for the model’s predictions using SHAP. Users can see which factors contribute most to obesity classification of their input data from the previous Predictive Analytics section. 
Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.
"""

#Import prediction from previous dashboard
# Load model
loaded_model = None 
prediction = None
user_data = None

if "prediction" in st.session_state and "loaded_model" in st.session_state and "user_data" in st.session_state:
    prediction = st.session_state.prediction
    loaded_model = st.session_state.loaded_model
    user_data = st.session_state.user_data
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(user_data)

    st.markdown('<div id="SHAP Summary Plot"></div>', unsafe_allow_html=True)
    st.write("## SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, user_data, show=False, plot_type="bar")
    st.pyplot(fig)

    st.markdown('<div id="SHAP Force Plot"></div>', unsafe_allow_html=True)
    st.write("## SHAP Force Plot")
    ## st.pyplot(shap.plots.force(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :] ,matplotlib=True))
    st_shap(shap.force_plot(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :]), height=200, width=1000)

else:
    st.info("Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.")

         




########################### 1


