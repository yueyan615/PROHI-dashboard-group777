import pandas as pd
import streamlit as st

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
    # layout="wide"
)


############################ SIDEBAR
### Logo
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  

# st.sidebar.markdown("# About ℹ️")


############################ MAIN BODY
# img_body = st.image("logo2.png", use_container_width = True )
"""
# About the Dataset 
"""

st.markdown("""
This dashboard uses data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

**Dataset name:** *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*  

**Description:**  
The dataset contains information collected from individuals regarding their **eating habits**, **physical condition**, and **lifestyle behaviors**.  
It was designed to predict obesity levels, classified into categories such as *Underweight, Normal Weight, Overweight, and Obese* based on WHO standards.  
""")

"""
# About Our Team
"""
st.markdown("""
This dashboard was created by **Group 777** as part of assignemtn of Project Management and Tools for Health Informatics (PROHI).
The team members are:
Weiqi Kong 

Yueyan Li 

Zsolt Fehér 

Christoffer Brändefors 

Naznin Akhtar
""")

# st.code("x = 2021")
import streamlit as st

st.set_page_config(page_title="About | Obesity Analytics", page_icon="ℹ️", layout="wide")
st.title("About")

st.markdown("""
## Project Overview
This dashboard walks through **Descriptive → Diagnostic → Predictive → Prescriptive (Explainability)** analytics for an obesity classification problem.
Training happens in Jupyter; the Streamlit app only performs **inference and SHAP explanations** using a pre-trained model file.

## Dataset
- **Source**: Public obesity dataset on eating habits and physical condition  
- **Size**: ~2,100 rows; ~17 columns (features + multi-class label)  
- **Regions**: Latin America (e.g., Mexico/Peru/Colombia)  
- **Imbalance Handling**: Oversampling (e.g., SMOTE) used to address class imbalance  
- **Features**: Diet, activity, hydration, alcohol, screen time, transport, plus sex/age/height/weight  
- **Limitations**: Contains synthetic samples; labels often based on BMI thresholds—interpretation is exploratory

## Dashboard Structure & Methods
- **Descriptive**: Summaries, distributions, pivots, and basic visuals with filters/bins  
- **Diagnostic**: Correlations, pair plots, group comparisons, and optional clustering  
- **Predictive**: Loads a **pre-trained** `.pkl` model for on-demand predictions (no training in the app)  
- **Prescriptive (SHAP)**: Single/batch SHAP with summary/force/waterfall plots and concise textual insights

## Team & Contact
- **Team**: Group 7 (replace if needed)  
- **Members**: *Add names here*  
- **Contact**: *Add shared or lead email here* (e.g., group7-project@university.edu)  
- **Project Docs**: See the team charter for problem statement, design process, risks, and milestones

## Timeline & Scope
- Semester project milestones: proposal → requirements → beta dashboard → final delivery  
- **Boundaries**: Educational use; no clinical validation or EHR integrations; predictions are not guaranteed; SHAP aids understanding but does not replace expert judgment

## References (examples)
- Public dataset record/paper describing variables and oversampling approach  
- Recent studies on obesity classification (including ensemble baselines)  
- Team project charter and internal documentation
""")


