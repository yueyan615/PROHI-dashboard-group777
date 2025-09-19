import pandas as pd
import streamlit as st

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
    # layout="wide"
)


############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
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


