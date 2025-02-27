import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Set pandas display options
pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None  # default='warn'

# Load data
overview = pd.read_excel('./for PPT Aspirational block program.xlsx', sheet_name='Overview of Blocks', header=4)
overview = overview.dropna(axis=0)

dfs = pd.read_excel('./for PPT Aspirational block program.xlsx', sheet_name=None, header=None)

def merge_header_rows(df):
    main_headings = df.iloc[5].tolist()
    subheadings = df.iloc[6].tolist()
    
    main_headings_filled = []
    last_main_heading = ""
    for heading in main_headings:
        if pd.notna(heading):
            last_main_heading = heading
        main_headings_filled.append(last_main_heading)
    
    column_names = [f"{main} - {sub}" if pd.notna(sub) else main for main, sub in zip(main_headings_filled, subheadings)]
    
    return df.iloc[7:].set_axis(column_names, axis=1)

dfs = {
    "1.Health": merge_header_rows(dfs["1.Health"]),
    "1b.WCD": merge_header_rows(dfs["1b.WCD"]),
    "2.Eduation": merge_header_rows(dfs["2.Eduation"]),
    "3.Agriculture and Allied Servic": merge_header_rows(dfs["3.Agriculture and Allied Servic"]),
    "4.Basic Development": merge_header_rows(dfs["4.Basic Development"]),
    "5.Social Sector": merge_header_rows(dfs["5.Social Sector"]),
}

health = dfs["1.Health"]
healthwcd = dfs["1b.WCD"]
edu = dfs["2.Eduation"]
agri = dfs["3.Agriculture and Allied Servic"]
basicinf = dfs["4.Basic Development"]
social = dfs["5.Social Sector"]

characters_to_remove = ['/', "'", '-', '\n']

for char in characters_to_remove:
    overview.columns = overview.columns.str.replace(char, '', regex=False)
    health.columns = health.columns.str.replace(char, '', regex=False)
    healthwcd.columns = healthwcd.columns.str.replace(char, '', regex=False)
    edu.columns = edu.columns.str.replace(char, '', regex=False)
    agri.columns = agri.columns.str.replace(char, '', regex=False)
    basicinf.columns = basicinf.columns.str.replace(char, '', regex=False)
    social.columns = social.columns.str.replace(char, '', regex=False)

# Load shapefiles
disshp = gpd.read_file('./districts_Maharashtra.shp')
blkshp = gpd.read_file('./blocks_Maharashtra.shp')
abpshp = gpd.read_file('./ABP_Maharashtra.shp')

overview = overview.rename(columns={'BLOCK LGD CODE':'Block code'})
abpshp = abpshp.rename(columns={'BLOCKCode':'Block code'})
abpshp = abpshp.merge(overview, on='Block code', how='left')
abpshp = abpshp.rename(columns={'S.NO.':'slno'})

# Merge dataframes
healthdf = health.merge(healthwcd, left_on='Theme  Health and Nutrition1.a  S.NO.', right_on='Theme  Health and Nutrition1.b  S.NO.', how='left')
healthdf = healthdf.drop(columns=['Theme  Health and Nutrition1.b  S.NO.', 'Theme  Health and Nutrition1.b  DISTRICT NAME', 'Theme  Health and Nutrition1.b  BLOCK LGD CODE', 'Theme  Health and Nutrition1.b  BLOCK NAME'])

def select_even_columns(df):
    even_cols = [i for i in range(0, len(df.columns), 2) if i != 2]
    new_df = df.iloc[:, even_cols].copy() 

    if 0 in even_cols:
        new_df.columns = ["slno"] + list(new_df.columns[1:])

    return new_df

dfns = [healthdf, edu, agri, basicinf, social]
df_names = ["healthdf", "edu", "agri", "basicinf", "social"]

filtered_dfs = {name + "1": select_even_columns(df) for name, df in zip(df_names, dfns)}
healthdf1, edu1, agri1, basicinf1, social1 = filtered_dfs.values()

weights = {
    "healthdf1": 0.30,
    "edu1": 0.30,
    "agri1": 0.20,
    "basicinf1": 0.15,
    "social1": 0.05
}

def sum_odd_columns_wa(df, name):
    odd_cols = [i for i in range(len(df.columns)) if i % 2 == 1]  
    df[name] = df.iloc[:, odd_cols].sum(axis=1)  
    return df

dfs_with_sums = {name: sum_odd_columns_wa(df, name) for name, df in filtered_dfs.items()}
healthdf1, edu1, agri1, basicinf1, social1 = dfs_with_sums.values()

healthdf1["wa_healthdf1"] = healthdf1["healthdf1"] * weights["healthdf1"]
edu1["wa_edu1"] = edu1["edu1"] * weights["edu1"]
agri1["wa_agri1"] = agri1["agri1"] * weights["agri1"]
basicinf1["wa_basicinf1"] = basicinf1["basicinf1"] * weights["basicinf1"]
social1["wa_social1"] = social1["social1"] * weights["social1"]

dfs_to_merge = [healthdf1, edu1, agri1, basicinf1, social1]
merged_df = abpshp.copy()  

for df in dfs_to_merge:
    merged_df = pd.merge(merged_df, df, on="slno", how="left")

merged_df = merged_df.sort_values(by='slno')

normdf = merged_df[['slno', 'SUB_DIST', 'DISTRICT', 'mh_distric', 'mh_sub_dis', 'Block code', 'geometry','wa_healthdf1','wa_edu1','wa_agri1','wa_basicinf1','wa_social1']]

wa_columns = [col for col in normdf.columns if col.startswith("wa_")]

for col in wa_columns:
    normdf.loc[:, f"std_{col}"] = (normdf[col] - normdf[col].mean()) / normdf[col].std()

# Streamlit App
st.title("Aspirational Block Program Dashboard")

# Create Folium map
st.subheader("Interactive Map: Improvement in Score from June to Sept24")
m = folium.Map(location=[19.7515, 75.7139], zoom_start=6)

# Add choropleth layer
choropleth = folium.Choropleth(
    geo_data=abpshp,
    name='choropleth',
    data=abpshp,
    columns=['Block code', 'Improvement in Score from June to Sept24'],
    key_on='feature.properties.Block code',
    fill_color='RdYlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Improvement Score'
).add_to(m)

# Add hover functionality
folium.features.GeoJson(
    abpshp,
    name='Labels',
    style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['SUB_DIST', 'DISTRICT', 'Improvement in Score from June to Sept24'],
        aliases=['Block Name:', 'District:', 'Improvement Score:'],
        localize=True
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display Folium map
folium_static(m)

# Block selection for spider chart
st.subheader("Select a Block for Detailed Analysis")
block_names = normdf['SUB_DIST'].unique()
selected_block = st.selectbox("Choose a block", block_names)

# Spider chart function
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
    ax.plot(angles, [0] * len(angles), color='k', linewidth=1.2, linestyle=':') 
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label_mapping[col] for col in cols], fontsize=12)
    ax.spines['polar'].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8)  
    ax.yaxis.set_tick_params(width=2)  
    ax.set_ylim(-2.5, 3)  
    plt.title(f'Radar Plot for {name}')  
    st.pyplot(fig)

# Display spider chart for selected block
st.subheader(f"Performance Analysis for {selected_block}")
plot_spider_chart(normdf, normdf[normdf['SUB_DIST'] == selected_block].index[0])


# Create ol dataframe
ol = merged_df.merge(normdf[['slno','std_wa_healthdf1', 'std_wa_edu1', 'std_wa_agri1', 'std_wa_basicinf1', 'std_wa_social1']], on='slno', how='left')

ol['compositescore'] = (ol['std_wa_healthdf1'] + ol['std_wa_edu1'] + ol['std_wa_agri1'] + ol['std_wa_basicinf1'] + ol['std_wa_social1']) / 5

ol = ol.sort_values(by='compositescore', ascending=False)

ol['comp_rank'] = ol['compositescore'].rank(method='dense', ascending=False)



# Merge abpshp with ol for hover info
abpshp_hover = abpshp.merge(ol[['Block code', 'compositescore', 'comp_rank']], on='Block code', how='left')


# Add hover functionality with new info
folium.features.GeoJson(
    abpshp_hover,
    name='Labels',
    style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['SUB_DIST', 'DISTRICT', 'Improvement in Score from June to Sept24', 'compositescore', 'comp_rank'],
        aliases=['Block Name:', 'District:', 'Improvement Score:', 'Composite Score:', 'Composite Rank:'],
        localize=True
    )
).add_to(m)

# Create filter for top/bottom blocks by composite rank
st.subheader("Select Blocks by Composite Rank")
filter_options = ['All', 'Top 10', 'Bottom 10']
selected_filter = st.selectbox("Choose Filter", filter_options)



# Filter data based on selected option
if selected_filter == 'Top 10':
    filtered_ol = ol.head(10)
elif selected_filter == 'Bottom 10':
    filtered_ol = ol.tail(10)
else:
    filtered_ol = ol


# Visualize filtered data
st.subheader(f"Composite Scores for {selected_filter} Blocks")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(filtered_ol['SUB_DIST'], filtered_ol['compositescore'])
ax.set_xlabel('Block Name')
ax.set_ylabel('Composite Score')
ax.set_title(f'Composite Scores for {selected_filter} Blocks')
ax.tick_params(axis='x', rotation=90)
st.pyplot(fig)



def plot_bar_chart(inpname, df, column_name):
    # Convert column to numeric if necessary
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # Drop rows with NaN in the specified column
    df = df.dropna(subset=[column_name])
    
    if df.empty:
        st.error("No valid data for plotting.")
        return
    
    # Extract input value
    inp = df.loc[df['SUB_DIST'] == inpname, column_name].values[0]
    
    # Find max and min values
    max_index = df[column_name].idxmax()
    max_val_name = df.loc[max_index, 'SUB_DIST']
    max_val = df[column_name].max()
    
    min_index = df[column_name].idxmin()
    min_val_name = df.loc[min_index, 'SUB_DIST']
    min_val = df[column_name].min()
    
    # Prepare data for plotting
    labels = [inpname, max_val_name, min_val_name]
    values = [inp, max_val, min_val]
    
    # Plot bar chart
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color='black', linestyle='--')  # Add axis line at y=0
    
    ax.set_xlabel('Blocks')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Comparison of {column_name} for {inpname}, Max, and Min')
    
    return fig



# Get unique block names and column names
block_names = ol['SUB_DIST'].unique()
column_names = ol.columns.tolist()

# Remove unwanted columns (e.g., 'SUB_DIST', 'DISTRICT', etc.)
column_names = [col for col in column_names if col not in ['SUB_DIST', 'DISTRICT']]

# Create input widgets
st.subheader("Compare Block Performance")
inpname = st.selectbox("Choose a Block", block_names)
column_name = st.selectbox("Choose a Column", column_names)

# Plot bar chart based on user input
fig = plot_bar_chart(inpname, ol, column_name)
st.pyplot(fig)



import matplotlib.pyplot as plt

def plot_rank_percentage(inpname, df, column_name):
    # Filter data by SUB_DIST
    filtered_df = df[df['SUB_DIST'] == inpname]
    
    if filtered_df.empty:
        st.error("No data found for the selected block.")
        return
    
    # Extract rank and percentage
    rank = filtered_df['comp_rank'].values[0]
    percentage = filtered_df[column_name].values[0]
    
    # Plot data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display rank as text
    ax1.axis('off')  # Turn off axis
    ax1.text(0.5, 0.5, f'Overall Rank: {rank}', ha='center', va='center', size=20)
    
    # Plot percentage in a donut plot
    ax2.pie([percentage, 100 - percentage], colors=['blue', 'gray'], autopct='%1.1f%%', startangle=90, 
            wedgeprops=dict(width=0.3, edgecolor='w'))
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title(f'Percentage for {column_name}')
    
    return fig

# Get unique block names and column names
block_names = ol['SUB_DIST'].unique()
column_names = ol.columns.tolist()

# Remove unwanted columns (e.g., 'SUB_DIST', 'DISTRICT', etc.)
column_names = [col for col in column_names if col not in ['SUB_DIST', 'DISTRICT', 'comp_rank']]

# Create input widgets with unique keys
st.subheader("View Rank and Percentage for a Block")
inpname = st.selectbox("Choose a Block", block_names, key='block_selector')
column_name = st.selectbox("Choose a Column", column_names, key='column_selector')

# Plot data based on user input
fig = plot_rank_percentage(inpname, ol, column_name)
st.pyplot(fig)






st.header("Hypertension Screening Trendline")

# Add a selectbox for block_name (values from column 3)
block_names = health.iloc[:, 3].unique()
selected_block = st.selectbox("Select Block", block_names)

# Filter the health dataframe based on the selected block
filtered_health = health[health.iloc[:, 3] == selected_block]

# Check if the filtered dataframe is empty
if filtered_health.empty:
    st.warning(f"No data available for the selected block: {selected_block}")
else:
    # Get the place name (from the first row of the filtered data)
    place_name = filtered_health.iloc[0, 3]

    # Get the values from columns 23, 21, and 20
    values = filtered_health.iloc[0, [23, 21, 20]]

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values.index, values.values, marker='o', linestyle='-', label="Values")

    # Formatting
    ax.set_title(f"Trendline of Percentage of person screened for Hypertension against targeted population in the Block for {place_name}")
    ax.set_xlabel("Time periods")
    ax.set_ylabel("Values")
    ax.set_xticks(values.index)
    ax.set_xticklabels(['Baseline', 'June 24', 'Sept 24'])  # Custom x-axis labels
    ax.legend()
    ax.grid(True)

    # Show the plot in Streamlit
    st.pyplot(fig)




