import streamlit as st
import plotly.figure_factory as ff
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

st.sidebar.markdown("# Descriptive Analytics 📊")


############################ MAIN BODY
""" # Descriptive Analytics 📊"""

"""
Add here some descriptive analytics with Widgets and Plots

### ⚠️ In-class exercise: Integrate a plot from plotly examples

🔗 Link: <https://plotly.com/python/scientific-charts/>
"""

# load the dataset
file_name = './assets/ObesityDataSet_cleaned.csv'
df = pd.read_csv(file_name)

########################### 1

### Create a CSV viewer
"""## 1. Obesity Dataset (cleaned)"""

# display the url
DATA_URL = "https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition"
st.caption(DATA_URL) 


# display the dataframe
st.dataframe(df, use_container_width=True)

# display the shape of dataframe
st.write(f"Total: **{df.shape[0]}** rows × **{df.shape[1]}** columns")


########################### 2
"""## 2. Random time series data"""

st.write("Streamlit supports a wide range of data visualizations, including [Plotly, Altair, and Bokeh charts](https://docs.streamlit.io/develop/api-reference/charts). 📊 And with over 20 input widgets, you can easily make your data interactive!")

all_users = ["Alice", "Bob", "Charly"]
with st.container(border=True):
    users = st.multiselect("Users", all_users, default=all_users)
    rolling_average = st.toggle("Rolling average")

np.random.seed(42)
data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
if rolling_average:
    data = data.rolling(7).mean().dropna()

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data, height=250)
tab2.dataframe(data, height=250, use_container_width=True)


########################### 3
"""## 3. Histogram chart"""

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

########################### 4