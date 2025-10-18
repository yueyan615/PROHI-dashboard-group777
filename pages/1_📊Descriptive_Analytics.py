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
file_name = './assets/ObesityDataSet_BMI.parquet'
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

    # Standardized as string labels
    if ptypes.is_bool_dtype(s):
        s = s.astype(str)  # "True"/"False"
        categories = ['False','True']  # Fixed order
    elif getattr(s.dtype, "name", "") == "category":
        categories = list(map(str, s.cat.categories))
        s = s.astype(str)
    else:
        # Other low-cardinality variables: by order of appearance
        s = s.astype(str)
        categories = list(pd.Index(s.dropna().unique()))

    # Generate color mapping
    color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}
    return s, categories, color_map







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
    sec = st.radio("Navigate on the page", ["Descriptive Analytics","Obesity Dataset", "Variable Distributions"], index=0)
    mapping = {"Descriptive Analytics": "Descriptive Analytics", "Obesity Dataset": "Obesity Dataset", "Variable Distributions": "Variable Distributions"}
    scroll_to(mapping[sec])



############################ MAIN BODY ############################
st.markdown('<div id="Descriptive Analytics"></div>', unsafe_allow_html=True)
""" # Descriptive Analytics"""

"""
This section provides an overview of the dataset. Users can explore basic summaries, pivot tables, and simple plots to understand the distribution of variables and obesity levels.
"""



########################### 1 ####################################
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
            var_type = "Categorical_Ordinal" if s.cat.ordered else "Categorical_Nominal"
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


########################### 2 ####################################
st.markdown('<div id="Variable Distributions"></div>', unsafe_allow_html=True)

st.markdown("## Variable Distributions")
st.markdown("This section lets you inspect individual variables: view overall distributions and compare distributions across obesity levels using interactive charts and tables.")

option = st.selectbox("**Select a variable to display**", list(df.columns), index=0)
TABLE_H2 = 340
if option:
    col = option

    # Determine the variable type
    if df[col].dtype == "bool" or df[col].nunique() <= 10:

        st.markdown(f"#### Overall {option} Distribution")

        # Standardized as string + Get categories and color mapping
        plot_s, categories, color_map = categorical_setup(df[col])

########################### 2.1 Overall Distribution ##############################
        # Overall count (using string column)
        overall_count = plot_s.value_counts(dropna=False).reindex(categories).reset_index()
        overall_count.columns = [col, "Count"]

        # Pie chart (unified category_orders and color_discrete_map)
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

        st.divider()



############################ 2.2 Distribution by Obesity Level ##############################
        # Grouped chart (excluding 'Obesity_level' and 'BMI')
        if col != 'Obesity_level' and col != 'BMI':
            st.markdown(f"#### {option} Distribution By Obesity Level")
            # Note: Use string column plot_s for grouping
            tmp = pd.DataFrame({
                'Obesity_level': df['Obesity_level'].astype(str),
                col: plot_s
            })
            count_df = tmp.groupby(['Obesity_level', col]).size().reset_index(name='Count')

            tab3, tab4 = st.tabs(["Chart", "Crosstab"])
            with tab3:
                # Calculate the percentage within each obesity level
                count_df_with_pct = count_df.copy()
                count_df_with_pct['Percentage'] = count_df_with_pct.groupby('Obesity_level')['Count'].transform(
                    lambda x: (x / x.sum() * 100).round(1)
                )
                count_df_with_pct['Label'] = count_df_with_pct['Count'].astype(str) + '<br>(' + count_df_with_pct['Percentage'].astype(str) + '%)'
                
                fig = px.bar(
                    count_df_with_pct,
                    x='Obesity_level',
                    y='Count',
                    color=col,
                    barmode='group',
                    category_orders={
                        'Obesity_level': OBESITY_ORDER,
                        col: categories
                    },
                    color_discrete_map=color_map,
                    text='Label'  # Use custom labels
                )
                fig.update_layout(
                    height=TABLE_H2,
                    legend_title_text="",
                    bargap=0.3,        # Control the gap between different groups (obesity levels)
                    bargroupgap=0.1    # Control the gap within the same group
                )
                
                # Set the hover information
                fig.update_traces(
                    textposition='outside',
                    textfont=dict(size=10),
                    hovertemplate='<b>%{fullData.name}</b><br>' +  # Show category name
                                'Obesity Level: %{x}<br>' +        # Show obesity level
                                'Count: %{y}<br>' +                # Show count
                                'Percentage: %{customdata}%<br>' + # Show percentage
                                '<extra></extra>',                 # Remove default trace box
                    customdata=count_df_with_pct['Percentage']       # Pass percentage data
                )

                st.plotly_chart(fig, use_container_width=True)
 
            with tab4:
                ctab = pd.crosstab(plot_s, df["Obesity_level"].astype(str)).reindex(index=categories, columns=OBESITY_ORDER).fillna(0).astype(int)
                st.dataframe(ctab, use_container_width=True, height=TABLE_H2)
            st.divider()
############################# 2.3 Each Category's Obesity Level Distribution Pie Charts #############################
            st.markdown(f"#### Obesity Level Distribution For Each {option} Category")
            # Pie chart 1: Show the obesity level distribution for each category
            # First row: first 4
            legend_cols_1 = st.columns(4)
            for i in range(4):
                if i < len(OBESITY_ORDER):
                    obesity_level = OBESITY_ORDER[i]
                    with legend_cols_1[i]:
                        color = PALETTE[i % len(PALETTE)]
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                            f'<div style="width: 12px; height: 12px; background-color: {color}; margin-right: 6px; border-radius: 2px;"></div>'
                            f'<span style="font-size: 11px;">{obesity_level}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Second row: last 3
            legend_cols_2 = st.columns([1, 1, 1, 1])  # 4 columns but only use the first 3 to keep it centered
            for i in range(3):
                idx = i + 4
                if idx < len(OBESITY_ORDER):
                    obesity_level = OBESITY_ORDER[idx]
                    with legend_cols_2[i]:
                        color = PALETTE[idx % len(PALETTE)]
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                            f'<div style="width: 12px; height: 12px; background-color: {color}; margin-right: 6px; border-radius: 2px;"></div>'
                            f'<span style="font-size: 11px;">{obesity_level}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Add spacing
            st.write("")

            # Draw pie charts for each category separately
            num_categories = len(categories)
            cols_per_row = min(3, num_categories)  # A maximum of 3 pie charts per line
            
            for i, category in enumerate(categories):
                if i % cols_per_row == 0:
                    # Create a new row and add spacing between each row
                    if i > 0:
                        st.write("")  # Add spacing
                    pie_cols = st.columns(cols_per_row, gap="medium")  # Add column spacing

                # Current category's obesity level distribution
                category_data = count_df[count_df[col] == category]
                
                with pie_cols[i % cols_per_row]:
                    if not category_data.empty:
                        fig_cat_pie = px.pie(
                            category_data,
                            names='Obesity_level',
                            values='Count',
                            title=f"{category}",
                            category_orders={'Obesity_level': OBESITY_ORDER},
                            color='Obesity_level',
                            color_discrete_map={lvl: PALETTE[i % len(PALETTE)] for i, lvl in enumerate(OBESITY_ORDER)},
                        )
                        # Add percentage and count labels to the pie chart
                        fig_cat_pie.update_traces(
                            texttemplate='%{value}<br>(%{percent})',  # Show count and percentage
                            textposition='inside',
                            textfont_size=10,
                            hovertemplate='<b>%{label}</b><br>' +      # Show obesity level name
                                        'Count: %{value}<br>' +       # Show count
                                        'Percentage: %{percent}<br>' +# Show percentage
                                        '<extra></extra>'             # Remove default trace box
                        )
                        fig_cat_pie.update_layout(
                            height=240,
                            showlegend=False,  # Each pie chart does not show legend, using shared legend
                            title_font_size=16,
                            # title_x=0.4,
                            margin=dict(t=35, b=15, l=15, r=15)
                        )
                        st.plotly_chart(fig_cat_pie, use_container_width=True)
                    else:
                        st.write(f"No data for {category}")
            


############################## 2.4 BMI Distribution by Each Category ##############################
            st.divider()
            st.markdown(f"#### BMI Distribution For Each {option} Category")

            # Simple handling: Use a stringified temporary column as the plotting column, and booleans can also share the color_map with categories.
            plot_s = df[col].astype(str)
            tmp_df = df.copy()
            tmp_df["_plot_col"] = plot_s

            fig_box = px.box(
                tmp_df,
                x="_plot_col",
                y="BMI",
                category_orders={"_plot_col": categories},
                color="_plot_col",
                color_discrete_map=color_map,
            )
            fig_box.update_xaxes(title_text=col)
            fig_box.update_layout(
                height=400,
                title="",
                title_font_size=14,
                legend_title_text="",
                margin=dict(t=50, b=50, l=50, r=50)
            )

            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                st.plotly_chart(fig_box, use_container_width=True)

            with c2:
                st.markdown("<br><b></b>", unsafe_allow_html=True)
                with st.container(height=TABLE_H2):
                    st.write("##### Mean BMI for each category")
                    # Calculate the mean using string columns, ensuring consistency between booleans and strings
                    for category in categories:
                        mean_bmi = df[plot_s == category]['BMI'].mean()
                        st.write(f"- {category}: **{mean_bmi:.2f}**")

                       



    else:
        # ====== Continuous variables: Unified main color as PALETTE[0] ======
        st.markdown(f"**Overall {option} Distribution**")
        category_options = list(df[option].unique())

        fig_overall = px.histogram(
            df,
            x=col,
            nbins=30,
        )
        # Histogram: Unified bar color
        fig_overall.update_traces(marker_color=PALETTE[0], opacity=0.8)  

        tab1, tab2 = st.tabs(["Chart", "Details"])
        with tab1:
            fig_overall.update_layout(height=TABLE_H2, width=800)
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2:
                st.plotly_chart(fig_overall, use_container_width=True)

        with tab2:
            summary_df = df[[col]].describe()
            st.dataframe(summary_df, use_container_width=True, height=TABLE_H2)
        # Box plot (grouped by obesity level): All boxes in the same color
        st.markdown(f"**{option} Distribution By Obesity Level**")
        fig = px.box(
            df,
            x='Obesity_level',
            y=col,
            category_orders={'Obesity_level': OBESITY_ORDER},
        )
        # Box plot: Unified line/fill/outlier colors
        fig.update_traces(
            line_color=PALETTE[0],      # Box line color
            # fillcolor=PALETTE[0],       # Box fill color
            marker_color=PALETTE[0],    # Outlier color
            # opacity=0.8,                # Moderate transparency for overlapping observation
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




    
