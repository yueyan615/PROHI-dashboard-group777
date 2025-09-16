import streamlit as st


st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="logo1.png",
)


### Logo
img1 = 'logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  
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


