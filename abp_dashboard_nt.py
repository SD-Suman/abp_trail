import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

# Load data
disshp = gpd.read_file('./districts_Maharashtra.shp')
abpshp = gpd.read_file('./ABP_Maharashtra.shp')
ol = pd.read_csv('./abp_data.csv')
health = pd.read_csv('./healthonly.csv')

# Merge data
oln = abpshp[['BLOCKCode', 'geometry']].merge(ol, left_on='BLOCKCode', right_on='Blockcode', how='left')

# Extract relevant columns for filtering
start_col = 'September Delta Ranking across 500 blocks'
filter_cols = oln.columns[oln.columns.get_loc(start_col):]

# Streamlit UI
st.title("Maharashtra ABP Data Visualization")
selected_columns = st.multiselect("Select columns to display on hover:", filter_cols, default=[filter_cols[0]])

# Normalize values for color mapping
cmap = cm.get_cmap('RdYlGn')
norm = colors.Normalize(vmin=oln[selected_columns[0]].min(), vmax=oln[selected_columns[0]].max())

# Create Folium map
m = folium.Map(location=[19.75, 75.71], zoom_start=6, tiles='cartodb positron')

# Add district boundaries with light grey fill
disshp_layer = folium.GeoJson(
    disshp,
    style_function=lambda feature: {'fillColor': 'lightgrey', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.6}
)
disshp_layer.add_to(m)

# Add block-level data
for _, row in oln.iterrows():
    color = colors.to_hex(cmap(norm(row[selected_columns[0]]))) if not pd.isnull(row[selected_columns[0]]) else "gray"
    folium.GeoJson(
        row['geometry'],
        style_function=lambda feature, color=color: {'fillColor': color, 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7},
        tooltip=folium.Tooltip(f"{row['SUB_DIST']}<br>{'<br>'.join([f'{col}: {row[col]}' for col in selected_columns])}")
    ).add_to(m)

# Display map
folium_static(m)

# Radar Chart Function
def plot_spider_chart(df, row_index):
    label_mapping = {
        'std_wa_healthdf1': 'Health',
        'std_wa_edu1': 'Education',
        'std_wa_agri1': 'Agriculture',
        'std_wa_basicinf1': 'Basic Infra',
        'std_wa_social1': 'Social'
    }
    
    cols = list(label_mapping.keys())
    values = df.loc[row_index, cols].values
    name = df.loc[row_index, 'SUB_DIST']  
    num_vars = len(cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label_mapping[col] for col in cols], fontsize=12)
    ax.set_ylim(-2.5, 3)
    plt.title(f'Radar Plot for {name}')
    st.pyplot(fig)

# User Selection for Radar Chart
st.header("Compare Two Blocks")
available_blocks = ol['SUB_DIST'].unique()
selected_blocks = st.multiselect("Select two blocks to compare:", available_blocks, default=available_blocks[:2])

if len(selected_blocks) == 2:
    col1, col2 = st.columns(2)
    with col1:
        plot_spider_chart(ol[ol['SUB_DIST'] == selected_blocks[0]], ol[ol['SUB_DIST'] == selected_blocks[0]].index[0])
    with col2:
        plot_spider_chart(ol[ol['SUB_DIST'] == selected_blocks[1]], ol[ol['SUB_DIST'] == selected_blocks[1]].index[0])

# Bar Chart Function
def plot_bar_chart(inpname, df, column_name):
    inp = df.loc[df['SUB_DIST'] == inpname, column_name].values[0]
    max_index = df[column_name].idxmax()
    max_val_name = df.loc[max_index, 'SUB_DIST']
    max_val = df[column_name].max()
    min_index = df[column_name].idxmin()
    min_val_name = df.loc[min_index, 'SUB_DIST']
    min_val = df[column_name].min()
    avg_val = df[column_name].mean()
    labels = [inpname, max_val_name, min_val_name]
    values = [inp, max_val, min_val]
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.axhline(avg_val, color='black', linestyle='--', label='Average')
    plt.xlabel('Blocks')
    plt.ylabel('Percentage')
    plt.title(f'Comparison of {column_name} for {inpname}, Max, and Min')
    plt.legend()
    st.pyplot(plt)

# User Selection for Bar Chart
st.header("Bar Chart Comparison")
selected_block = st.selectbox("Select a block:", available_blocks, index=0)
selected_column = st.selectbox("Select a column:", filter_cols, index=0)
plot_bar_chart(selected_block, oln, selected_column)

# # User Selection for Trendline Analysis
st.header("Trendline Analysis")

# Create a dropdown for selecting a place
available_places = health.iloc[:, 2].unique()
selected_place = st.selectbox("Select a place:", available_places)

# Filter data based on selection
filtered_data = health[health.iloc[:, 2] == selected_place]

if not filtered_data.empty:
    place_name = filtered_data.iloc[0, 2]  # Get the place name
    values = filtered_data.iloc[0, [12, 10, 9]]  # Get the values

    # Plot the trendline
    plt.figure(figsize=(8, 5))
    plt.plot(values.index, values.values, marker='o', linestyle='-', label="Values")
    plt.title(f"Trendline of Percentage of institutional deliveries against total reported deliveries for {place_name}")
    plt.xlabel("Time periods")
    plt.ylabel("Values")
    plt.xticks(values.index, ['Baseline', 'June 24', 'Sept 24'])  # Custom x-axis labels
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write("No data available for the selected place.")
