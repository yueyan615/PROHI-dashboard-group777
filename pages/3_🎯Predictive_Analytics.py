import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Predictive Analytics | Obesity Analytics",
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

# st.sidebar.caption("© 2025 Group 777 | Project Management and Tools for Health Informatics (PROHI)")



############################ MAIN BODY

""" # Predictive Analytics"""
"""
This section allows users to input new data and obtain predictions on obesity levels using a pre-trained machine learning model. The dashboard focuses on making predictions rather than training models.
"""



########################### 1
"""## Input Your Data Here"""

# """
# ⚠️ Add here some predictive analytics with Widgets and Plots
# """

# st.write("# Example of model prediction")

# # Load model
# pre_trained_model_path = "./assets/trained_model.pickle"
# loaded_model = None # This will be replaced by the trained model in the pickle 

# with open(pre_trained_model_path, "rb") as readFile:
#     loaded_model = pickle.load(readFile)


# # COLUMNS
# left_column, right_column = st.columns(2)

# user_data = []
# # Call Streamlit functions inside a "with" block to keep it in a column:
# with left_column:
#     length = st.slider("Sepal Length", min_value=4.0, max_value=9.0, value = 5.0)
# with right_column:
#     width = st.slider("Sepal Width", min_value=1.5, max_value=4.0, value = 3.0)

# if st.button('Predict!'):
#     user_data = [[length, width]]
#     prediction = loaded_model.predict(user_data)
#     st.write(f"The predicted value for data {user_data} is {prediction}")

# """
# # 
# ⚠️ Add some visualizations to help understanding what the predictions mean...
# """

##############################################################################


### Example of input widgets


with st.form("my_form"):
    # slider_val = st.slider("Form slider")
    # checkbox_val = st.checkbox("Form checkbox")

    gender = st.radio(
    "Gender", ("Male", "Female"),
    )

    age = st.number_input(
        "Age (year)",
        placeholder="Type a number...",
        value=None,
        # min_value=14,
        # max_value=61,
        format="%g"   # Do not force a fixed number of decimal places; display according to the actual significant digits.
    )
    
    st.write("The current number is ", age)


    height = st.number_input(
        "Height (m)",
        value=None,
        placeholder="Type a number...",
        format="%g"   # Do not force a fixed number of decimal places; display according to the actual significant digits.
    )
    if height is not None:
        st.write("The current number is ", height)
    else:
        st.write("The current number is None")



    weight = st.number_input(
        "Weight (kg)",
        value=None,
        placeholder="Type a number...",
        format="%g"   # Do not force a fixed number of decimal places; display according to the actual significant digits.
    )
    if weight is not None:
        st.write("The current number is ", weight)
    else:
        st.write("The current number is None")

    ##########################################

    q1 = st.selectbox(
        "Q1",
        ("0: xxxxxxxxxx", "1: xxxxxxxxxxxx", "2: xxxxxxxxxxxxxxx")
    )

    q2 = st.selectbox(
        "Q2",
        ("0: xxxxxxxxxx", "1: xxxxxxxxxxxx", "2: xxxxxxxxxxxxxxx")
    )

    q3 = st.selectbox(
        "Q3",
        ("0: xxxxxxxxxx", "1: xxxxxxxxxxxx", "2: xxxxxxxxxxxxxxx")
    )
    




    # Every form must have a submit button.
    pred = st.form_submit_button("Predict🎯")
    # if pred:
    #     st.write("slider", slider_val, "checkbox", checkbox_val)









######################################################################


