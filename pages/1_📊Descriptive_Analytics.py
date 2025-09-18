import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from pandas.api import types as ptypes

st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
    # layout="wide"
)


############################ SIDEBAR
### Logo
img1 = './img/logo_nb.png'
st.logo(img1, size= "large", icon_image=None)  

# st.sidebar.markdown("# Descriptive Analytics 📊")


############################ MAIN BODY
""" # Descriptive Analytics"""

"""
This section provides an overview of the dataset. Users can explore basic summaries, pivot tables, and simple plots to understand the distribution of variables and obesity levels.
"""
st.markdown("<br>", unsafe_allow_html=True)


# load the dataset
file_name = './assets/ObesityDataSet_cleaned.parquet'
df = pd.read_parquet(file_name)

########################### 1

### Create a CSV viewer
"""## Obesity Dataset"""

# display the url
DATA_URL = "https://archive.ics.uci.edu/dataset/544"
st.caption(f"🔗 {DATA_URL}" ) 


# display the dataframe
# st.dataframe(df, use_container_width=True)

# display the dataframe


tab1, tab2 = st.tabs(["Dataframe(cleaned)", "Detials"])
with tab1:
    st.dataframe(df, use_container_width=True)
# ...existing code...

with tab2:
    st.write(f"Total: **{df.shape[0]}** rows × **{df.shape[1]}** columns")

    # 自动生成列信息表（dtype / Variable Type / categories or numeric summary）
    rows = []
    for col in df.columns:
        s = df[col]
        # 推断变量类型
        if s.dtype.name == "category":
            var_type = "Ordinal" if s.cat.ordered else "Categorical"
            cats = list(map(str, s.cat.categories))
            n_cats = len(cats)
            preview = ", ".join(cats[:20] + (["..."] if n_cats > 20 else []))
            summary = f"{n_cats} categories: {preview}"
        elif ptypes.is_bool_dtype(s):
            var_type = "Boolean"
            cats = ["True", "False"]
            summary = "Categories: True, False"
        elif ptypes.is_numeric_dtype(s):
            var_type = "Numerical"
            try:
                mn = s.min()
                mx = s.max()
                mean = s.mean()
                summary = f"{mn:g} ~ {mx:g} (mean = {mean:g})"
            except Exception:
                summary = ""
        else:
            # 小基数的 object 当作分类处理
            if s.nunique(dropna=True) <= 20:
                var_type = "Categorical"
                unique_vals = list(map(str, s.dropna().unique().tolist()))
                n_cats = len(unique_vals)
                preview = ", ".join(unique_vals[:20] + (["..."] if n_cats > 20 else []))
                summary = f"{n_cats} categories: {preview}"
            else:
                var_type = "Other"
                summary = ""

        rows.append({
            "Column": col,
            "Variable_Type": var_type,
            "Summary": summary
        })

    var_summary_df = pd.DataFrame(rows)

    # st.subheader("Columns summary (Variable Type & Summary)")
    st.dataframe(var_summary_df, use_container_width=True, height=600)






st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)



########################### 2
"""## Explore Variables by Obesity Level"""


option = st.selectbox(
    "",
    list(df.columns),
    index=None,
    placeholder="Select Columns...",
)

st.markdown("<br>", unsafe_allow_html=True)

if option:
    col = option

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
        
        if getattr(df[col].dtype, "name", "") == "category":
            categories = list(df[col].cat.categories)
        else:
            categories = list(df[col].dropna().unique().astype(str))

        # 生成颜色映射（按 categories 顺序）
        palette = px.colors.qualitative.Plotly
        color_map = {str(cat): palette[i % len(palette)] for i, cat in enumerate(categories)}

        # 饼图：指定 category_orders 和 color_discrete_map
        overall_count = df[col].value_counts(dropna=False).reset_index()
        overall_count.columns = [col, "Count"]
        fig_overall = px.pie(
            overall_count,
            names=col,
            values="Count",
            category_orders={col: categories},
            color_discrete_map=color_map,
        )       


        tab1, tab2 = st.tabs(["Chart", "Table"])
        with tab1:
            st.plotly_chart(fig_overall, use_container_width=True)
        with tab2:
            st.dataframe(overall_count, use_container_width=True)

        # Categorical variables: bar chart by group
        st.markdown(f"**{option} Distribution by Obesity Level**")
        count_df = df.groupby(['Obesity_level', col]).size().reset_index(name='Count')
        count_df[col] = count_df[col].astype(str)
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
            st.plotly_chart(fig, use_container_width=False)
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





    
