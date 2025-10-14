import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Diagnostic | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_BMI_one_hot.parquet'
df = pd.read_parquet(file_name)

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
    sec = st.radio("Navigate on the page", ["Diagnostic Analytics","Heatmap"], index=0)
    mapping = {"Diagnostic Analytics": "Diagnostic Analytics", "Heatmap": "Heatmap"}
    scroll_to(mapping[sec])



############################ MAIN BODY
st.markdown('<div id="Diagnostic Analytics"></div>', unsafe_allow_html=True)
""" # Diagnostic Analytics"""

"""
This section helps users explore relationships between variables. Through correlations, statistical tests, and clustering, users can investigate potential factors that explain differences in obesity levels.
"""

########################### 0
"""## A Quick Look"""

container = st.container(border=True)
with container:
    col1, col2 = st.columns([1, 2], gap="large")
    with col2:
        two_options = st.multiselect(
            "Select exactly 2 features to see their correlation:", 
            df.columns.tolist(), 
            default=df.columns.tolist()[:2], 
            key="two_feature_selection",
            max_selections=2
            )

    with col1:
        if two_options and len(two_options) == 2:
            corr_two = df[two_options].corr()
            st.metric(f"Correlation Coefficient", f"{corr_two.iloc[0, 1]:.3f}")
        elif len(two_options) < 2:
            st.info("Please select exactly 2 features")
        else:
            st.warning("Please select only 2 features")

# Add explanation
st.info(f"ℹ️ The default method here for correlation calculation is Pearson")
st.divider()

########################### 1
st.markdown('<div id="Heatmap"></div>', unsafe_allow_html=True)
"""## Heatmap of Correlation Matrix"""

# ================== correlation elements pills  ===================
st.markdown("#### Customize Correlation Heatmap")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    option_map = {
        0: "pearson",
        1: "spearman",
        2: "kendall",
    }

    method_key = st.pills(
        "**Correlation Methods:**",
        options=option_map.keys(),
        format_func=lambda option: option_map[option].capitalize(),  # 显示时首字母大写
        selection_mode="single",
        default=0
    )

    # If no option is selected, use the default value.
    # if method_key is None:
    #     method_key = 0

    method = option_map[method_key]


    # Add explanation
    with st.expander("**How to choose?**"):
        st.markdown("""
            - **Pearson**: Most common; measures linear relationships; sensitive to outliers.  
            - **Spearman**: Rank-based; better for monotonic but non-linear relationships.  
            - **Kendall**: Concordance-based; more robust but slower to compute. 
        """)


with col2:
    options = df.columns.tolist()
    selected_features = st.pills(
        "**Features:**", 
        options, 
        selection_mode="multi",
        default=options,
        key="correlation_method"
    )
    # Add explanation
    st.info(f"ℹ️ Since Transportation_mode is an ordinal feature, it has been one-hot encoded to allow for better correlation analysis.")





# ================== heatmap  ===================
# Calculate the correlation matrix
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"#### Correlation Matrix ({method.capitalize()} method)")

corr = df[selected_features].corr(method=method)

custom_scale = [
    (0.0, "#0072b2"),   # 对应 zmin (-1) - 蓝色
    (0.5, "#ffffff"),   # 对应 0 - 白色
    (1.0, "#e69f00"),   # 对应 zmax (1) - 橙色
]

# Create a heatmap
fig_heatmap = px.imshow(
    corr,
    x=corr.columns,
    y=corr.columns,
    color_continuous_scale=custom_scale,
    zmin=-1,
    zmax=1,
    aspect="auto",
    text_auto=".2f",
    # title=f"Correlation Matrix ({method.capitalize()} method)"
)

# Update layout - enlarge the image
fig_heatmap.update_layout(
    height=max(700, 30 * len(corr.columns)),  
    width=max(700, 30 * len(corr.columns)),   
    xaxis_title="Features",
    yaxis_title="Features",
    font=dict(size=12),  
    margin=dict(l=120, r=120, t=50, b=120),  
)


# Display coefficient values on the heatmap
fig_heatmap.update_traces(
    text=np.round(corr.values, 2),  
    texttemplate="%{text}",         
    textfont={"size": 11, "color": "black"},  
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
)

# Add this line to force a square ratio
fig_heatmap.update_xaxes(scaleanchor="y", scaleratio=1)

# Customize hover information
fig_heatmap.update_traces(
    texttemplate="%{text}",
    textfont={"size": 8},
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
)

# Rotate x-axis labels for better readability
fig_heatmap.update_xaxes(tickangle=45)

# Show the heatmap
st.plotly_chart(fig_heatmap, use_container_width=True)



st.divider()



########################### 2 Display highly correlated feature pairs
st.markdown("#### High Correlation Feature Pairs")



# Calculate the correlation matrix using the features selected by the user
selected_corr = df[selected_features].corr(method=method)

# Create a mask to get the upper triangle of the correlation matrix, excluding the diagonal
mask = np.triu(np.ones_like(selected_corr, dtype=bool), k=1)
corr_masked = selected_corr.where(mask)

# Extract feature pairs with high correlation
high_corr_pairs = []

for i in range(len(corr_masked.columns)):
    for j in range(len(corr_masked.columns)):
        if not pd.isna(corr_masked.iloc[i, j]):
            high_corr_pairs.append({
                'Feature 1': corr_masked.columns[i],
                'Feature 2': corr_masked.columns[j],
                'Correlation': corr_masked.iloc[i, j]
            })

# Dynamically set the maximum value of the slider to the actual number of pairs
max_pairs = min(len(high_corr_pairs), 20)  # 最多显示20个

# 处理边界情况：当只有1个或0个配对时
if max_pairs == 0:
    st.warning("⚠️ No feature pairs available. Please select at least 2 features in the heatmap section.")
    num_pairs = 0
elif max_pairs == 1:
    st.info("Only 1 feature pair available from selected features, showing it below:")
    num_pairs = 1
else:
    # 只有当配对数量大于1时才显示slider
    num_pairs = st.slider("Choose number of pairs to display:", 1, max_pairs, min(10, max_pairs), key="num_pairs_slider")

# 只有当有配对数据时才处理DataFrame和图表
if len(high_corr_pairs) > 0:
    # Sort by absolute value and take the top num_pairs
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(num_pairs)

    # Create feature pairs with labels (shorten the labels to improve readability)
    high_corr_df['Feature_Pair'] = high_corr_df.apply(
        lambda row: f"{row['Feature 1'][:15]}{'...' if len(row['Feature 1']) > 15 else ''} ↔ {row['Feature 2'][:15]}{'...' if len(row['Feature 2']) > 15 else ''}", 
        axis=1
    )

    # Create a horizontal bar chart
    fig_bar = px.bar(
        high_corr_df, 
        x='Correlation',
        y='Feature_Pair',
        orientation='h',
        color='Correlation',
        color_continuous_scale=[
            (0.0, "#0072b2"),   
            (0.5, "#ffffff"),     
            (1.0, "#e69f00")    
        ],
        range_color=[-1, 1],
        text=[f"{corr:.3f}" for corr in high_corr_df['Correlation']],
        title="",
        hover_data={
            'Feature 1': True,
            'Feature 2': True, 
            'Correlation': ':.3f',
            'Feature_Pair': False
        }
    )

    # Update layout
    fig_bar.update_layout(
        height=max(400, 40 * len(high_corr_df)),
        yaxis=dict(
            autorange="reversed",
            title="Feature Pairs",
            tickfont=dict(size=10)
        ),
        xaxis=dict(
            range=[-1, 1],
            title="Correlation Coefficient",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        ),
        showlegend=False,
        margin=dict(l=250, r=150, t=50, b=50),  
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=True,
        coloraxis=dict(
            colorbar=dict(
                title=dict(text="Correlation", font=dict(size=12)), 
                tickfont=dict(size=10),
                len=1.0,
                x=1.05,
            )
        )
    )

    # text position outside the bars and customize hover information
    fig_bar.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='black'),
        hovertemplate='<b>%{customdata[0]}</b> ↔ <b>%{customdata[1]}</b><br>' +
                        'Correlation: %{x:.3f}<br>' +
                        '<extra></extra>',
        customdata=high_corr_df[['Feature 1', 'Feature 2']].values
    )

    # Add a vertical line at x=0
    fig_bar.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    # Show the chart
    col1, col2, col3 = st.columns([0.5, 20, 1])
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)



# 如果没有足够的特征来计算相关性，不显示任何内容


st.divider()
########################### 3 Summary of Correlation Statistics 
st.markdown("#### Correlation Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Max Positive Correlation", f"{corr_masked.max().max():.3f}")

with col2:
    st.metric("Max Negative Correlation", f"{corr_masked.min().min():.3f}")

with col3:
    avg_corr = corr_masked.abs().mean().mean()
    st.metric("Average |Correlation|", f"{avg_corr:.3f}")

with col4:
    high_corr_count = (corr_masked.abs() >= 0.5).sum().sum()
    st.metric("High Corr Pairs (≥0.5)", high_corr_count)

