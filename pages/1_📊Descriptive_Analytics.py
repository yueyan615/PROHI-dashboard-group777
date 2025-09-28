import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from pandas.api import types as ptypes
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Descriptive | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_cleaned.parquet'
df = pd.read_parquet(file_name)


# ====== Color palette & helpers ======
PALETTE = ["#0072b2","#e69f00","#009e73","#cc79a7","#f0e442","#d55e00","#56b4e9"]

OBESITY_ORDER = [
    'Insufficient_Weight','Normal_Weight','Overweight_Level_I',
    'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'
]

def categorical_setup(series: pd.Series):
    """Return (plot_series_str, categories_list, color_map)."""
    s = series.copy()

    # 标准化为字符串标签
    if ptypes.is_bool_dtype(s):
        s = s.astype(str)  # "True"/"False"
        categories = ['False','True']  # 固定顺序
    elif getattr(s.dtype, "name", "") == "category":
        categories = list(map(str, s.cat.categories))
        s = s.astype(str)
    else:
        # 其它低基数变量：按出现顺序
        s = s.astype(str)
        categories = list(pd.Index(s.dropna().unique()))

    # 生成颜色映射
    color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}
    return s, categories, color_map







############################ SIDEBAR
### Logo
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  


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
    sec = st.radio("Navigate on the page", ["Descriptive Analytics","Obesity Dataset", "Variable Distributions"], index=0)
    mapping = {"Descriptive Analytics": "Descriptive Analytics", "Obesity Dataset": "Obesity Dataset", "Variable Distributions": "Variable Distributions"}
    scroll_to(mapping[sec])



############################ MAIN BODY
st.markdown('<div id="Descriptive Analytics"></div>', unsafe_allow_html=True)
""" # Descriptive Analytics"""

"""
This section provides an overview of the dataset. Users can explore basic summaries, pivot tables, and simple plots to understand the distribution of variables and obesity levels.
"""



########################### 1
st.markdown('<div id="Obesity Dataset"></div>', unsafe_allow_html=True)
"""## Obesity Dataset"""
"""
This dataset has been cleaned and rounded where appropriate; floats in categorical variables have been converted to integers, and, to improve readability, both the column names and the category labels have been renamed.
"""

# display the dataframe
TABLE_H = 300

tab1, tab2 = st.tabs(["Dataframe(cleaned)", "Detials"])
with tab1:
    st.dataframe(df, use_container_width=True, height=TABLE_H)


with tab2:
    

    rows = []
    for col in df.columns:
        s = df[col]

        if s.dtype.name == "category":
            var_type = "Ordinal" if s.cat.ordered else "Categorical"
            cats = list(map(str, s.cat.categories))
            preview = ", ".join(cats[:20] + (["..."] if len(cats) > 20 else []))
            summary = preview

        elif ptypes.is_bool_dtype(s):
            var_type = "Boolean"
            summary = "Yes, No"

        elif ptypes.is_numeric_dtype(s):
            var_type = "Numerical"
            try:
                mn = s.min()
                mx = s.max()
                summary = f"{mn:g} ~ {mx:g}"
            except Exception:
                summary = ""



        rows.append({
            "Column": col,
            "Variable_Type": var_type,
            "Range / Categories": summary
        })

    var_summary_df = pd.DataFrame(rows)

    # st.subheader("Columns summary")
    st.dataframe(var_summary_df, use_container_width=True, height=TABLE_H)

# option = st.selectbox("**Select a feature to display**", list(df.columns), index=0)
st.write(f"Total: **{df.shape[0]}** rows × **{df.shape[1]}** columns")
st.divider()


########################### 2
st.markdown('<div id="Variable Distributions"></div>', unsafe_allow_html=True)

st.markdown("## Variable Distributions")
st.markdown("This section lets you inspect individual variables: view overall distributions and compare distributions across obesity levels using interactive charts and tables.")

option = st.selectbox("**Select a variable to display**", list(df.columns), index=0)
TABLE_H2 = 340
if option:
    col = option

    # Determine the variable type
    if df[col].dtype == "bool" or df[col].nunique() <= 10:

         # ====== 分类/布尔变量：统一配色与标签 ======
        st.markdown(f"**Overall {option} Distribution**")

        # 统一为字符串 + 取得类别与颜色映射
        plot_s, categories, color_map = categorical_setup(df[col])

        # Overall 计数（使用字符串列）
        overall_count = plot_s.value_counts(dropna=False).reindex(categories).reset_index()
        overall_count.columns = [col, "Count"]

        # 饼图（统一 category_orders 和 color_discrete_map）
        fig_overall = px.pie(
            overall_count,
            names=col,
            values="Count",
            category_orders={col: categories},
            color=col,
            color_discrete_map=color_map,
        )

        tab1, tab2 = st.tabs(["Chart", "Crosstab"])
        with tab1:
            fig_overall.update_layout(height=TABLE_H2)
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                st.plotly_chart(fig_overall, use_container_width=True)
        with tab2:

            total = overall_count["Count"].sum()
            st.dataframe(overall_count, use_container_width=True, height=TABLE_H2)



        # 分组图（除 'Obesity_level' 与 'BMI' 外）
        if col != 'Obesity_level' and col != 'BMI':
            st.markdown(f"**{option} Distribution by Obesity Level**")
            # 注意：分组时用字符串列 plot_s
            tmp = pd.DataFrame({
                'Obesity_level': df['Obesity_level'].astype(str),
                col: plot_s
            })
            count_df = tmp.groupby(['Obesity_level', col]).size().reset_index(name='Count')

            fig = px.bar(
                count_df,
                x='Obesity_level',
                y='Count',
                color=col,
                barmode='group',
                category_orders={
                    'Obesity_level': OBESITY_ORDER,
                    col: categories
                },
                color_discrete_map=color_map,
            )
            fig.update_layout(legend_title_text="")
            tab3, tab4 = st.tabs(["Chart", "Crosstab"])
            with tab3:
                fig.update_layout(height=TABLE_H2)
                c1, c2, c3 = st.columns([1, 6, 1])
                with c2:
                    st.plotly_chart(fig, use_container_width=True)
            with tab4:
                ctab = pd.crosstab(plot_s, df["Obesity_level"].astype(str)).reindex(index=categories, columns=OBESITY_ORDER).fillna(0).astype(int)
                st.dataframe(ctab, use_container_width=True, height=TABLE_H2)


    else:
        # ====== 连续变量：统一主色为 PALETTE[0] ======
        st.markdown(f"**Overall {option} Distribution**")
        fig_overall = px.histogram(
            df,
            x=col,
            nbins=30,
        )
        # 直方图：柱体颜色统一
        fig_overall.update_traces(marker_color=PALETTE[0], opacity=0.8)  # 或 marker=dict(color=PALETTE[0])

        tab1, tab2 = st.tabs(["Chart", "Details"])
        with tab1:
            fig_overall.update_layout(height=TABLE_H2, width=800)
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2:
                st.plotly_chart(fig_overall, use_container_width=True)

        with tab2:
            summary_df = df[[col]].describe()
            st.dataframe(summary_df, use_container_width=True, height=TABLE_H2)
        # 箱线图（按肥胖等级分组）：所有箱体同色
        st.markdown(f"**{option} Distribution by Obesity Level**")
        fig = px.box(
            df,
            x='Obesity_level',
            y=col,
            category_orders={'Obesity_level': OBESITY_ORDER},
        )
        # 箱线图：线条/填充/离群点统一
        fig.update_traces(
            line_color=PALETTE[0],      # 箱体线条颜色
            # fillcolor=PALETTE[0],       # 箱体填充颜色
            marker_color=PALETTE[0],    # 离群点颜色
            # opacity=0.8,                # 适度半透明，便于重叠观察
            selector=dict(type='box')
        )

        tab3, tab4 = st.tabs(["Chart", "Details"])
        with tab3:
            fig.update_layout(height=TABLE_H2, width=800)
            c1, c2, c3 = st.columns([1, 6, 1])
            with c2:
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            summary_by_group = df.groupby('Obesity_level')[col].describe().T.reindex(columns=OBESITY_ORDER)
            st.dataframe(summary_by_group, use_container_width=True, height=TABLE_H2)




    
