import streamlit as st

img1 = 'logo_nb.png'
img2 = 'logo2.png'
img3 = 'logo3.png'


### Sidebar
# with st.sidebar:

    # img2 = 'logo3.png'
    # img_body = st.image(img2, 
    #                     use_container_width = True
    #                     )
    
#     add_selectbox = st.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )
    
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )
   
    # st.page_link("DV.py", label="Home", icon="🏠")
    # st.page_link(".streamlit/pages/page_1.py", label="Page 1", icon="1️⃣")
    # st.page_link(".streamlit/pages/page_2.py", label="Page 2", icon="2️⃣")

   
#######################################



### Logo
img1 = 'logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  
img2 = "SU_large.png"
st.sidebar.image(img2)
#######################################




### Main body
# img_body = st.image(img_logo, use_container_width =True)
st.write("Hello ,let's learn how to build a streamlit app together")
st.title("This is the app title")
st.header("This is the header")
st.markdown("This is the markdown")
st.subheader("This is the subheader")
st.caption("This is the caption")
st.code("x = 2021")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')
#######################################


### Map
import pandas as pd
import streamlit as st
from numpy.random import default_rng as rng

df = pd.DataFrame(
    {
        "col1": rng(0).standard_normal(1000) / 50 + 37.76,
        "col2": rng(1).standard_normal(1000) / 50 + -122.4,
        "col3": rng(2).standard_normal(1000) * 100,
        "col4": rng(3).standard_normal((1000, 4)).tolist(),
    }
)

st.map(df, latitude="col1", longitude="col2", size="col3", color="col4")
#######################################