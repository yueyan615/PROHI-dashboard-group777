import streamlit as st
import pandas as pd


st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="logo1.png",
)


############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  

st.sidebar.markdown("# About ℹ️")


############################ MAIN BODY
# img_body = st.image("logo2.png", use_container_width = True )
"""
# About the Dataset 
"""
# load the dataset
file_name = './assets/ObesityDataSet_cleaned.csv'
df = pd.read_csv(file_name)

# display the dataframe
st.dataframe(df, use_container_width=True)

# display the shape of dataframe
st.write(f"Total: **{df.shape[0]}** rows × **{df.shape[1]}** columns")

"""
# About Our Team 
"""

# st.code("x = 2021")


