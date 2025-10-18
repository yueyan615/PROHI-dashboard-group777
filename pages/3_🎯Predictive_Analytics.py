import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt

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
    sec = st.radio("Navigate on the page", ["Predictive Analytics","Input Your Data Here", "Prediction Results"], index=0)
    mapping = {"Predictive Analytics": "Predictive Analytics", "Input Your Data Here": "Input Your Data Here", "Prediction Results": "Prediction Results"}
    scroll_to(mapping[sec])



############################ MAIN BODY
st.markdown('<div id="Predictive Analytics"></div>', unsafe_allow_html=True)
""" # Predictive Analytics"""
"""
This section allows users to input new data and obtain predictions on obesity levels using a pre-trained machine learning model. The dashboard focuses on making predictions rather than training models.
"""



########################### 1
st.markdown('<div id="Input Your Data Here"></div>', unsafe_allow_html=True)
"""## Input Your Data Here"""

# Load model
pre_trained_model_path = "./assets/xgb_model.pkl"
loaded_model = None 

with open(pre_trained_model_path, "rb") as readFile:
    loaded_model = pickle.load(readFile)


with st.form("my_form"):
    # slider_val = st.slider("Form slider")
    # checkbox_val = st.checkbox("Form checkbox")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        gender = st.radio(
        "**Gender**", ("Male", "Female"),
        )

        age = st.number_input(
            "**Age (year)**",
            placeholder="Type a number...",
            value=18,
            min_value=18,
            max_value=100,
            format="%i"   # Do not force a fixed number of decimal places; display according to the actual significant digits.
        )

        fam_hist = st.radio(
        "**Has a family member suffered or suffers from overweight?**", ("Yes", "No"),
        )

        high_cal = st.radio(
        "**Do you eat high caloric food frequently?**", ("Yes", "No"),
        )

        vegie = st.radio(
        "**Do you usually eat vegetables in your meals?**", ("Never", "Sometimes", "Always"),
        )

        meals = st.radio(
            "**How many main meals do you have daily?**", ("Between 1 and 2", "Three", "More than three"),
        )

        snacks = st.radio(
            "**Do you eat food between meals?**", ("No", "Sometimes", "Frequently", "Always"),
        )

        smoke = st.radio(
            "**Do you smoke?**", ("Yes", "No"),
        )

    with c2:
        water = st.radio(
            "**How many liters of water do you drink daily?**", ("Less than a liter", "Between 1 and 2 L", "More than 2 L"),
        )
    
        monitor = st.radio(
            "**Do you monitor the calories you eat daily?**", ("Yes", "No"),
        )

        pysical = st.radio(
            "**How often do you have physical activity?**", ("I do not", "1 or 2 days", "2 or 4 days", "4 or 5 days"),
        )

        screen = st.radio(
            "**How much time do you use technological devices such as cell phone, videogames, television, computer and others?**", ("0-2 hours", "3-5 hours", "More than 5 hours"),
        )

        alcohol = st.radio(
            "**How often do you drink alcohol?**", ("I do not drink", "Sometimes", "Frequently", "Always"),
        )

        transport = st.radio(
            "**Which transportation do you usually use?**", ("Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"),
        )

  

    #now we need to convert the inputs into the format that the model expects
    # Create a DataFrame with the user input
    pred = st.form_submit_button("Predict🎯")

st.markdown('<div id="Prediction Results"></div>', unsafe_allow_html=True)
st.write("## Prediction Results")
if pred:
    user_data = pd.DataFrame({"Gender": [gender],
                            "Age": [age],
                            "Family_history_overweight": [fam_hist],
                            "High_caloric_food": [high_cal],
                            "Veggie_consumption_freq": [vegie],
                            "Main_meals_count": [meals],
                            "Food_between_meals_freq": [snacks],
                            "Smokes": [smoke],
                            "Water_consumption": [water],
                            "Monitors_calories": [monitor],
                            "Physical_activity": [pysical],
                            "Screen_time": [screen],
                            "Alcohol_consumption_freq": [alcohol],
                            "Transportation_mode": [transport]
                            })
    

    # after submit show the user data
    st.write("#### Your Input")
    st.dataframe(user_data)

    # Convert categorical variables to numerical using the method from prediction
    def preprocess_input(X):
        # Gender   
        X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
        X["Family_history_overweight"] = X["Family_history_overweight"].map({'Yes': 1, 'No': 0})

        X["High_caloric_food"] = X["High_caloric_food"].map({'Yes': 1, 'No': 0})
        X["Smokes"] = X["Smokes"].map({'Yes': 1, 'No': 0})
        X["Monitors_calories"] = X["Monitors_calories"].map({'Yes': 1, 'No': 0})

        X["Alcohol_consumption_freq"] = X["Alcohol_consumption_freq"].map({"Sometimes": 1, "Frequently": 2, "I do not drink": 0, "Always": 3})

        X["Physical_activity"] = X["Physical_activity"].map({"I do not": 0, "1 or 2 days": 2, "2 or 4 days": 3, "4 or 5 days": 4})

        X["Veggie_consumption_freq"] = X["Veggie_consumption_freq"].map({"Sometimes": 1, "Frequently": 2, "Never": 0, "Always": 3})

        X["Main_meals_count"] = X["Main_meals_count"].map({"Between 1 and 2": 0, "Three" : 1, "More than three": 2})

        X["Food_between_meals_freq"] = X["Food_between_meals_freq"].map({"Sometimes": 1, "Frequently": 2, "No": 0, "Always": 3})

        X["Water_consumption"] = X["Water_consumption"].map({"Less than a liter": 0, "Between 1 and 2 L": 1, "More than 2 L": 2})

        X["Screen_time"] = X["Screen_time"].map({"0-2 hours": 0, "3-5 hours": 1, "More than 5 hours": 2})

        X["Transportation_mode"] = X["Transportation_mode"].map({"Automobile": 5, "Public_Transportation": 4, "Public Transportation": 4, "Motorbike": 3, "Bike": 2, "Walking": 1})

        return X

    df_copy = preprocess_input(user_data.copy())

    prediction = loaded_model.predict(df_copy)

    obesity_levels = {
        0: 'Insufficient Weight',
        1: 'Normal Weight',
        2: 'Overweight Level I',
        3: 'Overweight Level II',
        4: 'Obesity Type I',
        5: 'Obesity Type II',
        6: 'Obesity Type III'
    }

    st.write("<br>", unsafe_allow_html=True)
    # The predicted obesity level with probability for the given input data is :
    st.write("#### Your Result")
    # st.write(f"**{obesity_levels[prediction[0]]}**")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; margin:10px 0;">
          <div style="border:2px solid #0072b2; background-color:#f7fbfd; padding:14px 20px; border-radius:8px; min-width:320px; text-align:center;">
            <span style="font-size:28px; color:#0072b2; font-weight:700; line-height:1.2">{obesity_levels[prediction[0]]}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    #show the probability for each class
    prediction_proba = loaded_model.predict_proba(df_copy)
    proba_df = pd.DataFrame(prediction_proba, columns=[obesity_levels[i] for i in range(len(obesity_levels))])

    st.write("<br>", unsafe_allow_html=True)
    st.write("#### Probabilities For Each Obesity Level")
    # Display the probabilities as a bar chart in ordered from lowest to highest

    st.bar_chart(proba_df.T, horizontal=True, color="#0072b2")

    
    st.session_state.prediction = prediction

    
    st.session_state.loaded_model = loaded_model
    
    
    st.session_state.user_data = df_copy

else:
    st.info("Fill in the form and click the Predict🎯 button to see the prediction results.")



######################################################################


