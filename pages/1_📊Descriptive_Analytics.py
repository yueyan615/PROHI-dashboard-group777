import streamlit as st
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
)


############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  

# st.sidebar.markdown("# Descriptive Analytics 📊")


############################ MAIN BODY
""" # Descriptive Analytics"""

"""
Description ...
"""
st.markdown("<br>", unsafe_allow_html=True)


# load the dataset
file_name = './assets/ObesityDataSet_cleaned.csv'
df = pd.read_csv(file_name)

########################### 1

### Create a CSV viewer
"""## Obesity Dataset (cleaned)"""

# display the url
DATA_URL = "https://archive.ics.uci.edu/dataset/544"
st.caption(f"🔗 {DATA_URL}" ) 


# display the dataframe
st.dataframe(df, use_container_width=True)

# display the shape of dataframe
st.write(f"Total: **{df.shape[0]}** rows × **{df.shape[1]}** columns")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ########################### 2
# """## 2. Random time series data"""

# st.write("Streamlit supports a wide range of data visualizations, including [Plotly, Altair, and Bokeh charts](https://docs.streamlit.io/develop/api-reference/charts). 📊 And with over 20 input widgets, you can easily make your data interactive!")

# all_users = ["Alice", "Bob", "Charly"]
# with st.container(border=True):
#     users = st.multiselect("Users", all_users, default=all_users)
#     rolling_average = st.toggle("Rolling average")

# np.random.seed(42)
# data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
# if rolling_average:
#     data = data.rolling(7).mean().dropna()

# tab1, tab2 = st.tabs(["Chart", "Dataframe"])
# tab1.line_chart(data, height=250)
# tab2.dataframe(data, height=250, use_container_width=True)


# ########################### 3
# """## 3. Histogram chart"""

# # Add histogram data
# x1 = np.random.randn(200) - 2
# x2 = np.random.randn(200)
# x3 = np.random.randn(200) + 2

# # Group data together
# hist_data = [x1, x2, x3]

# group_labels = ['Group 1', 'Group 2', 'Group 3']

# # Create distplot with custom bin_size
# fig = ff.create_distplot(
#         hist_data, group_labels, bin_size=[.1, .25, .5])

# # Plot!
# st.plotly_chart(fig, use_container_width=True)

########################### 2
"""## Explore Variables by Obesity Level"""

# Add category data
option_map = {
    "Gender": "Gender",
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "Family history of obesity": "Family_history_overweight",
    "High caloric food consumption": "High_caloric_food",
    "Frequency of vegetable intake": "Veggie_consumption_freq",
    "Number of primary meals": "Main_meals_count",
    "Consumption of food": "Food_between_meals_freq",
    "Smoking habit": "Smokes",
    "Water consumption per day": "Water_consumption",
    "Tracking calorie consumption": "Monitors_calories",
    "Frequency of physical activity": "Physical_activity",
    "Time spent on electronic gadgets": "Screen_time",
    "Alcohol consumption": "Alcohol_consumption_freq",
    "Type of transportation used": "Transportation_mode"
}

option = st.selectbox(
    "",
    list(option_map.keys()),
    index=None,
    placeholder="Select Columns...",
)

st.markdown("<br>", unsafe_allow_html=True)

if option:
    col = option_map[option]

    # Determine the variable type
    if df[col].dtype == "bool" or df[col].nunique() <= 10:

         # Categorical variable: Overall distribution
        st.markdown(f"**Overall {option} Distribution**")
        overall_count = df[col].value_counts(dropna=False).reset_index()
        overall_count.columns = [col, 'Count']
        fig_overall = px.pie(
            overall_count,
            names=col,
            values='Count',
            # title=f"Overall {option} Distribution"
        )
        tab1, tab2 = st.tabs(["Chart", "Table"])
        with tab1:
            st.plotly_chart(fig_overall, use_container_width=True)
        with tab2:
            st.dataframe(overall_count, use_container_width=True)

        # Categorical variables: bar chart by group
        st.markdown(f"**{option} Distribution by Obesity Level**")
        count_df = df.groupby(['Obesity_level', col]).size().reset_index(name='Count')
        fig = px.bar(
            count_df,
            x='Obesity_level',
            y='Count',
            color=col,
            barmode='group',
            category_orders={'Obesity_level': [
                'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
            ]},
            # title=f"{option} Distribution by Obesity Level"
        )
        tab3, tab4 = st.tabs(["Chart", "Table"])
        with tab3:
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.dataframe(count_df, use_container_width=True)
    

    else:
        # Continuous variable: Overall distribution
        st.markdown(f"**Overall {option} Distribution**")
        fig_overall = px.histogram(
            df,
            x=col,
            nbins=30,
            # title=f"Overall {option} Distribution"
        )
        tab1, tab2 = st.tabs(["Chart", "Table"])
        with tab1:
            st.plotly_chart(fig_overall, use_container_width=True)
        with tab2:
            st.dataframe(df[[col]], use_container_width=True)

        # Continuous variable: Box plot
        st.markdown(f"**{option} Distribution by Obesity Level**")
        fig = px.box(
            df,
            x='Obesity_level',
            y=col,
            category_orders={'Obesity_level': [
                'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
            ]},
            # title=f"{option} Distribution by Obesity Level"
        )
        tab3, tab4 = st.tabs(["Chart", "Table"])
        with tab3:
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.dataframe(df[[col, 'Obesity_level']], use_container_width=True)









    
