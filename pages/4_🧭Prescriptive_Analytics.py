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
    shap.summary_plot(shap_values, user_data, max_display=14,  plot_size=[10,4], class_names= ['Insufficient Weight','Normal Weight','Overweight Level I','Overweight Level II', 'Obesity Type I','Obesity Type II', 'Obesity Type III'] , show=False, plot_type="bar")
    st.pyplot(fig)
    st.write("The SHAP summary plot above shows the average impact of each feature on the model's predictions across all classes. Features are ranked by their importance, with the most influential features at the top. The length of each bar indicates the magnitude of the feature's contribution to the prediction, averaged over all samples.")

    st.divider()
    
    st.markdown('<div id="SHAP Force Plot"></div>', unsafe_allow_html=True)
    st.write("## SHAP Force Plot")
    ## st.pyplot(shap.plots.force(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :] ,matplotlib=True))
    st_shap(shap.force_plot(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :]), height=200, width=965)
    st.write("This plot provides a detailed explanation of the model's prediction for the given specific input data. It visualizes how each feature contributes to pushing the prediction from the base value (the average model output) to the final predicted value for the input. Features that increase the prediction are shown in red, while those that decrease it are shown in blue. The width of each arrow represents the magnitude of the feature's impact on the prediction")

else:
    st.warning("Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.")

         


