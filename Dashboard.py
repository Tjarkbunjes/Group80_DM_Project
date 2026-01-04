"""
The AIAI Strategic Explorer - Deliverable 2
Professional Customer Segmentation Dashboard for Executive Stakeholders

Architecture:
- Left Sidebar: Targeting Parameters (Behavioral & Demographic Filters)
- Main Area: Strategic Visualizations (3D Universe, Persona Profiling, Export)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="AIAI Strategic Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS with Green Palette (Light Theme) ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar styling - clean white background */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        width: 400px !important;
        min-width: 400px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 400px !important;
    }

    /* Sidebar text and labels */
    [data-testid="stSidebar"] label {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }

    [data-testid="stSidebar"] p {
        color: #6b7280 !important;
        font-size: 14px !important;
    }

    /* Sidebar general text */
    [data-testid="stSidebar"] {
        font-size: 15px !important;
    }

    /* Sidebar multiselect text */
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        font-size: 15px !important;
    }

    /* Sidebar input text */
    [data-testid="stSidebar"] input {
        font-size: 15px !important;
    }

    /* Checkbox styling - GREEN - Multiple approaches for compatibility */
    [data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: #45AF28 !important;
        cursor: pointer !important;
    }

    /* For browsers that don't support accent-color */
    [data-testid="stSidebar"] input[type="checkbox"]:checked {
        background-color: #45AF28 !important;
        border-color: #45AF28 !important;
    }

    /* Streamlit specific checkbox styling */
    [data-testid="stSidebar"] input[type="checkbox"]:checked::before {
        background-color: #45AF28 !important;
    }

    /* Alternative Streamlit checkbox selectors */
    [data-testid="stCheckbox"] input[type="checkbox"] {
        accent-color: #45AF28 !important;
    }

    [data-testid="stCheckbox"] input[type="checkbox"]:checked {
        background-color: #45AF28 !important;
        border-color: #45AF28 !important;
    }

    /* The checkmark itself */
    [data-testid="stCheckbox"] span[data-testid="stMarkdownContainer"] {
        color: #374151 !important;
    }

    /* Checkbox container styling */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        color: #374151 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        color: #374151 !important;
    }

    /* Sidebar expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #00622D !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        background-color: #f9fafb !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] > div:last-child {
        background-color: #ffffff !important;
        padding: 15px !important;
    }

    /* Sidebar headers h2 */
    [data-testid="stSidebar"] h2 {
        color: #10b981 !important;
        font-weight: 700 !important;
        font-size: 22px !important;
    }

    /* Multiselect and input styling */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
        color: #374151 !important;
    }

    [data-testid="stSidebar"] input:not([type="checkbox"]) {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
        color: #374151 !important;
    }

    /* Force multiselect container to stack vertically with dropdown on top */
    .stMultiSelect [data-baseweb="select"] > div:first-child {
        flex-direction: column-reverse !important;
        align-items: stretch !important;
    }

    /* Style multiselect tags to be full width */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: transparent !important;
        border: 1px solid #d1d5db !important;
        color: #374151 !important;
        padding: 4px 8px !important;
        margin: 2px 0 !important;
        font-size: 13px !important;
        width: 100% !important;
        display: flex !important;
        justify-content: space-between !important;
    }

    /* X button on tags */
    .stMultiSelect [data-baseweb="tag"] span[role="presentation"] {
        color: #6b7280 !important;
    }

    .stMultiSelect [data-baseweb="tag"]:hover {
        background-color: #f3f4f6 !important;
    }

    /* Slider thumb (handle) - green */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #10b981 !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Slider range values (min/max numbers) - black */
    [data-testid="stSidebar"] .stSlider label {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stSlider p {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stSlider div {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stSlider span {
        color: #000000 !important;
    }

    /* Remove all backgrounds from slider container */
    .stSlider > div {
        background-color: transparent !important;
    }

    .stSlider > div > div {
        background-color: transparent !important;
    }

    /* The actual track/rail - make it black, not red */
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
        background-color: #000000 !important;
    }

    div[data-baseweb="slider"] > div > div > div {
        background-color: #000000 !important;
    }

    /* Target the inner track element specifically */
    [data-baseweb="slider"] [role="presentation"] {
        background-color: #000000 !important;
    }

    /* Override any red color */
    .stSlider * {
        background-color: transparent !important;
    }

    /* But keep the track black */
    .stSlider [data-baseweb="slider"] > div > div > div {
        background-color: #000000 !important;
    }

    /* And the thumb green */
    .stSlider [role="slider"] {
        background-color: #10b981 !important;
    }

    /* Button styling */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #10b981 !important;
        border: 1px solid #10b981 !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #f0fdf4 !important;
        border-color: #059669 !important;
        color: #059669 !important;
    }

    /* Primary button for Select All */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #059669 !important;
    }

    /* Headers */
    h1 {
        color: #047857 !important;
        font-weight: 700 !important;
    }

    h2, h3, h4 {
        color: #065f46 !important;
    }

    /* Metrics cards */
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #047857 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    [data-testid="stMetricDelta"] {
        color: #059669 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 20px 0 10px 0;
    }

    /* Data source footer */
    .data-source {
        background-color: #d1fae5;
        color: #065f46;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 0.85rem;
        margin-top: 20px;
    }

    /* Cluster badge */
    .cluster-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Custom Color Palette (from Clustering Code) ---
CUSTOM_HEX = [
    "#00411E", "#00622D", "#00823C", "#45AF28", "#6BCF5D", "#D5E6D0", "#212121", "#313131", "#595959", "#909090"
]

# --- Behavioral Cluster Configuration (Group 80 Green Palette) ---
CLUSTER_CONFIG = {
    0: {
        'name': 'Disengaged Solo',
        'color': CUSTOM_HEX[8],  # #595959 - Gray for low engagement
        'desc': 'Low engagement, infrequent redemption, solo travelers.'
    },
    1: {
        'name': 'Business Commuters',
        'color': CUSTOM_HEX[1],  # #00622D - Dark green
        'desc': 'Regular solo business travelers, moderate engagement.'
    },
    2: {
        'name': 'Engaged Loyalists',
        'color': CUSTOM_HEX[3],  # #45AF28 - Bright green - High value
        'desc': 'Active, consistent, high redemption, loyal base.'
    },
    3: {
        'name': 'Family Travelers',
        'color': CUSTOM_HEX[2],  # #00823C - Medium green
        'desc': 'High companion ratio, vacation-oriented patterns.'
    },
    4: {
        'name': 'Explorers',
        'color': CUSTOM_HEX[4],  # #6BCF5D - Light green
        'desc': 'High distance variability, adventurous patterns.'
    }
}

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load and prepare customer segmentation data."""
    df = pd.read_csv('data/clustering_data/customer_segmentation_profiles.csv')

    # Handle missing values
    df['Income'] = df['Income'].fillna(0)
    df['Education'] = df['Education'].fillna('Unknown')
    df['City'] = df['City'].fillna('Unknown')
    df['Province or State'] = df['Province or State'].fillna('Unknown')
    df['fm_segment_fg1'] = df['fm_segment_fg1'].fillna('')

    # Add behavioral cluster names
    df['cluster_name'] = df['Behavioral_Cluster'].map(lambda x: CLUSTER_CONFIG.get(x, {}).get('name', f'Cluster {x}'))

    # Add demographic cluster names
    demographic_cluster_mapping = {
        0: 'Common Regions, Higher-Income Higher-Education',
        1: 'Lower-Income Lower-Education',
        2: 'Rare Regions, Higher-Education'
    }
    df['demographic_cluster_name'] = df['Demographic_Cluster'].map(demographic_cluster_mapping)

    # Create income bin names
    def bin_income(income):
        if income <= 20000:
            return 'Low Income (≤$20k)'
        elif income <= 70000:
            return 'Medium Income ($20k-$70k)'
        else:
            return 'High Income (>$70k)'

    df['Income_Bin_Name'] = df['Income'].apply(bin_income)

    return df

# --- Visualization Functions ---

def create_3d_universe(df: pd.DataFrame) -> go.Figure:
    """Create the 3D scatter plot - The Universe visualization."""

    fig = go.Figure()

    for cluster_id in sorted(df['Behavioral_Cluster'].unique()):
        cluster_df = df[df['Behavioral_Cluster'] == cluster_id]
        cluster_info = CLUSTER_CONFIG.get(cluster_id, {})
        cluster_name = cluster_info.get('name', f'Cluster {cluster_id}')
        cluster_color = cluster_info.get('color', '#6b7280')

        fig.add_trace(go.Scatter3d(
            x=cluster_df['pca_1'],
            y=cluster_df['pca_2'],
            z=cluster_df['pca_3'],
            mode='markers',
            name=cluster_name,
            marker=dict(
                size=5,
                color=cluster_color,
                opacity=0.6,
                line=dict(width=0)
            ),
            text=[f"<b>Loyalty# {row['Loyalty#']}</b><br>"
                  f"<b style='color: {cluster_color};'>{cluster_name}</b><br>"
                  f"━━━━━━━━━━━━<br>"
                  f"<b>Behavioral Metrics:</b><br>"
                  f"• Redemption: {row['redemption_frequency']:.3f}<br>"
                  f"• Companion Ratio: {row['companion_flight_ratio']:.3f}<br>"
                  f"• Flight Regularity: {row['flight_regularity']:.3f}<br>"
                  f"• Distance Variability: {row['distance_variability']:.3f}<br>"
                  f"━━━━━━━━━━━━<br>"
                  f"<b>Demographics:</b><br>"
                  f"• Province: {row['Province or State']}<br>"
                  f"• City: {row['City']}<br>"
                  f"• FSA: {row['FSA']}<br>"
                  f"• Gender: {row['Gender']}<br>"
                  f"• Education: {row['Education']}<br>"
                  f"• Location Code: {row['Location Code']}<br>"
                  f"• Income: ${row['Income']:,.0f}<br>"
                  f"• Marital Status: {row['Marital Status']}"
                  for _, row in cluster_df.iterrows()],
            hoverinfo='text',
            showlegend=True
        ))

    # Equal axis ranges
    all_points = pd.concat([df['pca_1'], df['pca_2'], df['pca_3']])
    axis_range = [all_points.min(), all_points.max()]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="<b>PC1: Engagement Axis</b>", font=dict(color='#047857', size=12)),
                backgroundcolor='#f9fafb',
                gridcolor='#d1d5eb',
                showbackground=True,
                range=axis_range
            ),
            yaxis=dict(
                title=dict(text="<b>PC2: Travel Pattern Axis</b>", font=dict(color='#047857', size=12)),
                backgroundcolor='#f9fafb',
                gridcolor='#d1d5eb',
                showbackground=True,
                range=axis_range
            ),
            zaxis=dict(
                title=dict(text="<b>PC3: Secondary Pattern Axis</b>", font=dict(color='#047857', size=12)),
                backgroundcolor='#f9fafb',
                gridcolor='#d1d5eb',
                showbackground=True,
                range=axis_range
            ),
            bgcolor='#ffffff',
            aspectmode='cube'
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='#1f2937'),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(color='#1f2937', size=13),
            title=dict(text="<b>Customer Segments</b>", font=dict(size=14, color='#1f2937')),
            x=1.0,
            y=0.9
        ),
        margin=dict(l=0, r=0, t=0, b=100),
        height=600,
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    )

    return fig


def create_persona_radar(df: pd.DataFrame, population_df: pd.DataFrame) -> go.Figure:
    """Create radar chart comparing selection to population baseline."""

    features = ['redemption_frequency', 'companion_flight_ratio', 'flight_regularity', 'distance_variability']
    labels = ['Redemption<br>Frequency', 'Companion<br>Ratio', 'Flight<br>Regularity', 'Distance<br>Variability']

    # Calculate means
    selected_means = [df[f].mean() for f in features]
    population_means = [population_df[f].mean() for f in features]

    # Normalize to 0-1
    normalized_selected = []
    normalized_population = []

    for i, f in enumerate(features):
        min_val = population_df[f].min()
        max_val = population_df[f].max()
        range_val = max_val - min_val if max_val > min_val else 1

        norm_sel = (selected_means[i] - min_val) / range_val
        norm_pop = (population_means[i] - min_val) / range_val

        normalized_selected.append(max(0, min(1, norm_sel)))
        normalized_population.append(max(0, min(1, norm_pop)))

    # Close the polygon
    normalized_selected.append(normalized_selected[0])
    normalized_population.append(normalized_population[0])
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    # Population baseline
    fig.add_trace(go.Scatterpolar(
        r=normalized_population,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(107, 114, 128, 0.2)',
        line=dict(color='#6b7280', width=2, dash='dash'),
        name='Population Baseline'
    ))

    # Selected segment
    fig.add_trace(go.Scatterpolar(
        r=normalized_selected,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.3)',
        line=dict(color='#10b981', width=3),
        name='Selected Customers'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#f9fafb',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#d1fae5',
                tickfont=dict(color='#047857', size=9),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '25', '50', '75', '100']
            ),
            angularaxis=dict(
                gridcolor='#d1fae5',
                tickfont=dict(color='#065f46', size=10)
            )
        ),
        paper_bgcolor='#ffffff',
        font=dict(color='#1f2937'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(color='#1f2937')
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )

    return fig


def create_demographic_split(df: pd.DataFrame, attribute: str) -> go.Figure:
    """Create stacked bar chart for demographic distribution."""

    # Count by cluster and attribute
    cross_tab = pd.crosstab(df['cluster_name'], df[attribute], normalize='index') * 100

    fig = go.Figure()

    # Use custom green palette for demographic bars
    colors_demo = [CUSTOM_HEX[3], CUSTOM_HEX[2], CUSTOM_HEX[1], CUSTOM_HEX[4], CUSTOM_HEX[5]]

    for i, attr_val in enumerate(cross_tab.columns):
        fig.add_trace(go.Bar(
            y=cross_tab.index,
            x=cross_tab[attr_val],
            name=str(attr_val),
            orientation='h',
            marker_color=colors_demo[i % len(colors_demo)],
            text=[f"{val:.1f}%" for val in cross_tab[attr_val]],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f9fafb',
        font=dict(color='#1f2937'),
        xaxis=dict(
            title=dict(text=f'<b>Distribution (%)</b>', font=dict(color='#047857')),
            gridcolor='#d1d5db',
            tickfont=dict(color='#065f46'),
            range=[0, 100]
        ),
        yaxis=dict(
            title=dict(text='<b>Customer Segment</b>', font=dict(color='#047857')),
            tickfont=dict(color='#1f2937'),
            showgrid=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(color='#1f2937', size=10),
            title=dict(text=f'<b>{attribute}</b>', font=dict(color='#1f2937', size=11))
        ),
        margin=dict(l=10, r=10, t=80, b=40),
        height=400
    )

    return fig


def create_fm_matrix_combined(df: pd.DataFrame) -> go.Figure:
    """Create combined FM Matrix scatter plot with both focus groups."""

    # Determine which focus group column to use (prefer fg1, fallback to fg2)
    fg_column = None
    tier_column = None

    if 'fm_segment_fg1' in df.columns and df['fm_segment_fg1'].notna().any():
        fg_column = 'fm_segment_fg1'
        tier_column = 'fm_tier_fg1'
    elif 'fm_segment_fg2' in df.columns and df['fm_segment_fg2'].notna().any():
        fg_column = 'fm_segment_fg2'
        tier_column = 'fm_tier_fg2'

    if fg_column is None:
        # Return empty figure if no FM data
        fig = go.Figure()
        fig.update_layout(
            title="FM Matrix data not available",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f9fafb',
            font=dict(color='#1f2937')
        )
        return fig

    # Filter to only rows with FM segment data
    df_with_fm = df[df[fg_column].notna()].copy()

    # Calculate medians and 90th percentiles
    freq_median = df_with_fm['Frequency'].median()
    mon_median = df_with_fm['Monetary'].median()
    freq_p90 = df_with_fm['Frequency'].quantile(0.90)
    mon_p90 = df_with_fm['Monetary'].quantile(0.90)

    # Define colors using custom palette
    segment_colors = {
        'Champions': CUSTOM_HEX[3],      # #45AF28 - Bright green
        'Frequent Flyer': CUSTOM_HEX[1], # #00622D - Dark green
        'Premium Occasional': CUSTOM_HEX[2], # #00823C - Medium green
        'At Risk': CUSTOM_HEX[4]         # #6BCF5D - Light green
    }
    elite_color = CUSTOM_HEX[0]  # #00411E - Darkest green for Elite

    fig = go.Figure()

    # Add quadrant backgrounds
    max_freq = df_with_fm['Frequency'].max() * 1.05
    max_mon = df_with_fm['Monetary'].max() * 1.05

    # Champions quadrant highlight (using Champions color)
    fig.add_shape(type="rect",
        x0=freq_median, x1=max_freq, y0=mon_median, y1=max_mon,
        fillcolor='rgba(69, 175, 40, 0.08)', line=dict(width=0), layer='below')

    # Elite zone highlight (using Elite color)
    fig.add_shape(type="rect",
        x0=freq_p90, x1=max_freq, y0=mon_p90, y1=max_mon,
        fillcolor='rgba(0, 65, 30, 0.15)', line=dict(width=0), layer='below')

    # Add median reference lines
    fig.add_hline(y=mon_median, line_dash="dash", line_color="#313131", line_width=2, opacity=0.7)
    fig.add_vline(x=freq_median, line_dash="dash", line_color="#313131", line_width=2, opacity=0.7)

    # Plot segments
    for segment in ['At Risk', 'Premium Occasional', 'Frequent Flyer', 'Champions']:
        segment_data = df_with_fm[df_with_fm[fg_column] == segment]

        if segment == 'Champions':
            # Separate Elite from Champions
            elite_data = segment_data[segment_data[tier_column] == 'Elite']
            non_elite_data = segment_data[segment_data[tier_column] != 'Elite']

            # Plot non-Elite Champions
            if len(non_elite_data) > 0:
                fig.add_trace(go.Scatter(
                    x=non_elite_data['Frequency'],
                    y=non_elite_data['Monetary'],
                    mode='markers',
                    name=f'Champions (n={len(non_elite_data):,})',
                    marker=dict(
                        size=8,
                        color=segment_colors[segment],
                        opacity=0.6,
                        line=dict(color='white', width=0.5)
                    ),
                    hovertemplate='<b>%{text}</b><br>Frequency: %{x:.2f}<br>Monetary: %{y:.2f}<extra></extra>',
                    text=[f"Customer {row['Loyalty#']}" for _, row in non_elite_data.iterrows()]
                ))

            # Plot Elite with distinct marker
            if len(elite_data) > 0:
                fig.add_trace(go.Scatter(
                    x=elite_data['Frequency'],
                    y=elite_data['Monetary'],
                    mode='markers',
                    name=f'Elite (n={len(elite_data):,})',
                    marker=dict(
                        size=10,
                        color=elite_color,
                        opacity=0.8,
                        symbol='diamond',
                        line=dict(color='white', width=0.8)
                    ),
                    hovertemplate='<b>%{text}</b><br>Frequency: %{x:.2f}<br>Monetary: %{y:.2f}<extra></extra>',
                    text=[f"Customer {row['Loyalty#']}" for _, row in elite_data.iterrows()]
                ))
        else:
            if len(segment_data) > 0:
                fig.add_trace(go.Scatter(
                    x=segment_data['Frequency'],
                    y=segment_data['Monetary'],
                    mode='markers',
                    name=f'{segment} (n={len(segment_data):,})',
                    marker=dict(
                        size=8,
                        color=segment_colors[segment],
                        opacity=0.6,
                        line=dict(color='white', width=0.5)
                    ),
                    hovertemplate='<b>%{text}</b><br>Frequency: %{x:.2f}<br>Monetary: %{y:.2f}<extra></extra>',
                    text=[f"Customer {row['Loyalty#']}" for _, row in segment_data.iterrows()]
                ))

    # Calculate label positions in corners of quadrants
    min_freq = df_with_fm['Frequency'].min()
    min_mon = df_with_fm['Monetary'].min()

    # Position labels in corners
    # Champions - bottom left corner of its quadrant
    champions_x = freq_median * 1.05
    champions_y = mon_median * 1.05

    # Premium Occasional - top left corner
    premium_x = min_freq * 1.1
    premium_y = max_mon * 0.95

    # Frequent Flyer - bottom right corner
    frequent_x = max_freq * 0.95
    frequent_y = min_mon * 1.1

    # At Risk - bottom left corner
    atrisk_x = min_freq * 1.1
    atrisk_y = min_mon * 1.1

    # Elite - top right corner of Champions quadrant
    elite_x = max_freq * 0.95;
    elite_y = max_mon * 0.95;

    # Add quadrant labels as annotations
    fig.add_annotation(x=champions_x, y=champions_y,
        text="Champions", showarrow=False,
        font=dict(size=14, color='#00411E', family="Arial Black"),
        xanchor='left', yanchor='bottom')

    fig.add_annotation(x=premium_x, y=premium_y,
        text="Premium Occasional", showarrow=False,
        font=dict(size=14, color='#00411E', family="Arial Black"),
        xanchor='left', yanchor='top')

    fig.add_annotation(x=frequent_x, y=frequent_y,
        text="Frequent Flyer", showarrow=False,
        font=dict(size=14, color='#00411E', family="Arial Black"),
        xanchor='right', yanchor='bottom')

    fig.add_annotation(x=atrisk_x, y=atrisk_y,
        text="At Risk", showarrow=False,
        font=dict(size=14, color='#00411E', family="Arial Black"),
        xanchor='left', yanchor='bottom')

    # Add Elite label
    fig.add_annotation(x=elite_x, y=elite_y,
        text="Elite", showarrow=False,
        font=dict(size=13, color='#00411E', family="Arial Black"),
        xanchor='right', yanchor='top')

    fig.update_layout(
        title=dict(
            text=f'FM Matrix: Value-Based Segmentation<br><sub>n={len(df_with_fm):,} customers</sub>',
            font=dict(color='#047857', size=14)
        ),
        xaxis=dict(
            title=dict(text='<b>Frequency (Flights per Active Month)</b>', font=dict(color='#047857')),
            gridcolor='#d1d5db',
            tickfont=dict(color='#065f46'),
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='<b>Monetary (Distance per Active Month)</b>', font=dict(color='#047857')),
            gridcolor='#d1d5db',
            tickfont=dict(color='#065f46'),
            showgrid=True,
            zeroline=False
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f9fafb',
        font=dict(color='#1f2937'),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(color='#1f2937', size=9)
        ),
        hovermode='closest',
        height=500
    )

    return fig


def create_segment_migration_sankey(df_selected: pd.DataFrame, df_population: pd.DataFrame) -> go.Figure:
    """
    Create Sankey diagram showing how selected customers are distributed across
    behavioral clusters and FM segments compared to the overall population.
    """
    
    # Determine which FM column to use
    fg_column = None
    if 'fm_segment_fg1' in df_selected.columns and df_selected['fm_segment_fg1'].notna().any():
        fg_column = 'fm_segment_fg1'
    elif 'fm_segment_fg2' in df_selected.columns and df_selected['fm_segment_fg2'].notna().any():
        fg_column = 'fm_segment_fg2'
    
    if fg_column is None:
        # Return empty figure if no FM data
        fig = go.Figure()
        fig.update_layout(
            title="FM segment data not available",
            paper_bgcolor='#ffffff',
            height=400
        )
        return fig
    
    # Filter to customers with FM segments
    df_sel_fm = df_selected[df_selected[fg_column].notna()].copy()
    df_pop_fm = df_population[df_population[fg_column].notna()].copy()
    
    # Create flow data: Behavioral Cluster -> FM Segment
    cluster_to_fm_selected = df_sel_fm.groupby(['cluster_name', fg_column]).size()
    cluster_to_fm_population = df_pop_fm.groupby(['cluster_name', fg_column]).size()
    
    # Create nodes
    clusters = sorted(df_pop_fm['cluster_name'].unique())
    fm_segments = sorted(df_pop_fm[fg_column].unique())
    
    # Node list: [clusters, fm_segments]
    node_labels = clusters + fm_segments
    
    # Create indices
    cluster_indices = {name: i for i, name in enumerate(clusters)}
    fm_indices = {name: i + len(clusters) for i, name in enumerate(fm_segments)}
    
    # Build links for selected customers
    sources_sel = []
    targets_sel = []
    values_sel = []
    
    for (cluster, fm_seg), count in cluster_to_fm_selected.items():
        sources_sel.append(cluster_indices[cluster])
        targets_sel.append(fm_indices[fm_seg])
        values_sel.append(count)
    
    # Build links for population (for comparison values)
    population_flows = {}
    for (cluster, fm_seg), count in cluster_to_fm_population.items():
        population_flows[(cluster, fm_seg)] = count
    
    # Calculate percentage representation
    hover_texts = []
    for i in range(len(sources_sel)):
        cluster_idx = sources_sel[i]
        fm_idx = targets_sel[i]
        cluster_name = clusters[cluster_idx]
        fm_name = fm_segments[fm_idx - len(clusters)]
        
        sel_count = values_sel[i]
        pop_count = population_flows.get((cluster_name, fm_name), 0)
        
        if pop_count > 0:
            percentage = (sel_count / pop_count) * 100
            hover_texts.append(
                f"{cluster_name} → {fm_name}<br>"
                f"Selected: {sel_count:,} customers<br>"
                f"Population: {pop_count:,} customers<br>"
                f"Representation: {percentage:.1f}%"
            )
        else:
            hover_texts.append(
                f"{cluster_name} → {fm_name}<br>"
                f"Selected: {sel_count:,} customers"
            )
    
    # Define colors for links (based on behavioral cluster)
    link_colors = []
    for src_idx in sources_sel:
        cluster_id = [k for k, v in CLUSTER_CONFIG.items() 
                     if v['name'] == clusters[src_idx]][0]
        cluster_color = CLUSTER_CONFIG[cluster_id]['color']
        # Add transparency
        if cluster_color.startswith('#'):
            rgb = tuple(int(cluster_color[i:i+2], 16) for i in (1, 3, 5))
            link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)')
        else:
            link_colors.append(cluster_color)
    
    # Node colors
    node_colors = []
    for cluster in clusters:
        cluster_id = [k for k, v in CLUSTER_CONFIG.items() if v['name'] == cluster][0]
        node_colors.append(CLUSTER_CONFIG[cluster_id]['color'])
    
    # FM segment colors using custom palette
    fm_colors = [CUSTOM_HEX[3], CUSTOM_HEX[2], CUSTOM_HEX[1], CUSTOM_HEX[4]]
    for i in range(len(fm_segments)):
        node_colors.append(fm_colors[i % len(fm_colors)])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=node_labels,
            color=node_colors,
            customdata=[f"n={df_sel_fm[df_sel_fm['cluster_name']==c].shape[0]:,}" if c in clusters
                       else f"n={df_sel_fm[df_sel_fm[fg_column]==c].shape[0]:,}" 
                       for c in node_labels],
            hovertemplate='%{label}<br>%{customdata}<extra></extra>'
        ),
        link=dict(
            source=sources_sel,
            target=targets_sel,
            value=values_sel,
            color=link_colors,
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Customer Journey: Behavioral Cluster → FM Segment<br><sub>Selected: {len(df_sel_fm):,} | Population: {len(df_pop_fm):,}</sub>",
            font=dict(color='#047857', size=14)
        ),
        font=dict(size=11, color='#1f2937'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f9fafb',
        height=500,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    return fig


# --- Main App ---

def main():
    # Load data
    try:
        df_full = load_data()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return

    # ==================== SIDEBAR: FILTERS ====================
    with st.sidebar:
        st.markdown("## Filters")

        # --- VALUE-BASED FILTERS ---
        with st.expander("Value-Based Filters", expanded=False):
            # Frequency Range
            freq_range = st.slider(
                "Frequency (Flights per Active Month)",
                min_value=float(df_full['Frequency'].min()),
                max_value=float(df_full['Frequency'].max()),
                value=(float(df_full['Frequency'].min()), float(df_full['Frequency'].max())),
                step=0.1
            )

            # Monetary Range
            monetary_range = st.slider(
                "Monetary (Distance per Active Month)",
                min_value=float(df_full['Monetary'].min()),
                max_value=float(df_full['Monetary'].max()),
                value=(float(df_full['Monetary'].min()), float(df_full['Monetary'].max())),
                step=10.0
            )

        # --- BEHAVIORAL FILTERS ---
        with st.expander("Behavioral Filters", expanded=True):

            # Customer Segments - Using nested expander for dropdown effect
            with st.expander("**Customer Segments**", expanded=False):
                segment_options = [CLUSTER_CONFIG[i]['name'] for i in sorted(CLUSTER_CONFIG.keys())]
                selected_segments = []
                for segment in segment_options:
                    if st.checkbox(segment, value=True, key=f"seg_{segment}"):
                        selected_segments.append(segment)

            # Redemption Frequency
            redemption_range = st.slider(
                "Redemption Frequency",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.01,
                format="%.2f"
            )

            # Flight Regularity
            flight_reg_range = st.slider(
                "Flight Regularity",
                min_value=float(df_full['flight_regularity'].min()),
                max_value=float(df_full['flight_regularity'].max()),
                value=(float(df_full['flight_regularity'].min()), float(df_full['flight_regularity'].max())),
                step=0.01
            )

            # Companion Flight Ratio
            companion_range = st.slider(
                "Companion Flight Ratio",
                min_value=float(df_full['companion_flight_ratio'].min()),
                max_value=float(df_full['companion_flight_ratio'].max()),
                value=(float(df_full['companion_flight_ratio'].min()), float(df_full['companion_flight_ratio'].max())),
                step=0.01
            )

            # Distance Variability
            distance_var_range = st.slider(
                "Distance Variability",
                min_value=float(df_full['distance_variability'].min()),
                max_value=float(df_full['distance_variability'].max()),
                value=(float(df_full['distance_variability'].min()), float(df_full['distance_variability'].max())),
                step=0.01
            )

        # --- DEMOGRAPHIC FILTERS ---
        with st.expander("Demographic Filters", expanded=False):
            # Province
            with st.expander("**Province**", expanded=False):
                province_options = sorted([p for p in df_full['Province or State'].unique() if pd.notna(p)])
                selected_provinces = []
                for province in province_options:
                    if st.checkbox(province, value=True, key=f"prov_{province}"):
                        selected_provinces.append(province)

            # City
            with st.expander("**City**", expanded=False):
                city_options = sorted([c for c in df_full['City'].unique() if pd.notna(c) and c != 'Unknown'])
                selected_cities = []
                for city in city_options:
                    if st.checkbox(city, value=True, key=f"city_{city}"):
                        selected_cities.append(city)

            # FSA
            with st.expander("**FSA (Forward Sortation Area)**", expanded=False):
                fsa_options = sorted([f for f in df_full['FSA'].unique() if pd.notna(f)])
                selected_fsa = []
                for fsa in fsa_options:
                    if st.checkbox(fsa, value=True, key=f"fsa_{fsa}"):
                        selected_fsa.append(fsa)

            # Gender
            with st.expander("**Gender**", expanded=False):
                gender_options = sorted([g for g in df_full['Gender'].unique() if pd.notna(g)])
                selected_gender = []
                for gender in gender_options:
                    if st.checkbox(gender, value=True, key=f"gender_{gender}"):
                        selected_gender.append(gender)

            # Education Level
            with st.expander("**Education Level**", expanded=False):
                education_options = sorted([e for e in df_full['Education'].unique() if pd.notna(e) and e != 'Unknown'])
                selected_education = []
                for education in education_options:
                    if st.checkbox(education, value=True, key=f"edu_{education}"):
                        selected_education.append(education)

            # Location Code
            with st.expander("**Location Code**", expanded=False):
                location_options = sorted([l for l in df_full['Location Code'].unique() if pd.notna(l)])
                selected_location = []
                for location in location_options:
                    if st.checkbox(location, value=True, key=f"loc_{location}"):
                        selected_location.append(location)

        st.divider()

    # ==================== APPLY FILTERS ====================
    df_filtered = df_full.copy()

    # Apply behavioral filters
    selected_cluster_ids = [k for k, v in CLUSTER_CONFIG.items() if v['name'] in selected_segments]
    df_filtered = df_filtered[df_filtered['Behavioral_Cluster'].isin(selected_cluster_ids)]

    df_filtered = df_filtered[
        (df_filtered['redemption_frequency'] >= redemption_range[0]) &
        (df_filtered['redemption_frequency'] <= redemption_range[1]) &
        (df_filtered['flight_regularity'] >= flight_reg_range[0]) &
        (df_filtered['flight_regularity'] <= flight_reg_range[1]) &
        (df_filtered['companion_flight_ratio'] >= companion_range[0]) &
        (df_filtered['companion_flight_ratio'] <= companion_range[1]) &
        (df_filtered['distance_variability'] >= distance_var_range[0]) &
        (df_filtered['distance_variability'] <= distance_var_range[1])
    ]

    # Apply demographic filters
    df_filtered = df_filtered[df_filtered['Province or State'].isin(selected_provinces)]
    df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]
    df_filtered = df_filtered[df_filtered['FSA'].isin(selected_fsa)]
    df_filtered = df_filtered[df_filtered['Gender'].isin(selected_gender)]
    df_filtered = df_filtered[df_filtered['Education'].isin(selected_education)]
    df_filtered = df_filtered[df_filtered['Location Code'].isin(selected_location)]

    # Apply value-based filters
    df_filtered = df_filtered[
        (df_filtered['Frequency'] >= freq_range[0]) &
        (df_filtered['Frequency'] <= freq_range[1]) &
        (df_filtered['Monetary'] >= monetary_range[0]) &
        (df_filtered['Monetary'] <= monetary_range[1])
    ]

    # ==================== MAIN HEADER ====================
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2.5rem; margin: 0;'>AIAI Customer Segmentation Strategy</h1>
        <p style='color: #059669; font-size: 1.1rem; margin-top: 10px;'>Behavioral Clustering & Value Analysis | Group 80</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==================== ROW 1: 3D VISUALIZATION ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>3D Customer Landscape</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        fig_3d = create_3d_universe(df_filtered)
        st.plotly_chart(fig_3d, width='stretch', config={'displaylogo': False})
    else:
        st.warning("No customers match the current filters. Please adjust your selection.")

    st.markdown("---")

    # ==================== ROW 2: FM MATRIX SEGMENTATION ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>FM Matrix: Value-Based Segmentation</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        fig_fm = create_fm_matrix_combined(df_filtered)
        st.plotly_chart(fig_fm, width='stretch', config={'displaylogo': False})
    else:
        st.warning("No customers match the current filters. Please adjust your selection.")

    st.markdown("---")

    # ==================== ROW 3: COMPARATIVE ANALYSIS ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>Comparative Analysis: Selected vs Population</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        persona_col1, persona_col2 = st.columns(2)

        with persona_col1:
            st.markdown("#### Segment Migration Flow")
            fig_sankey = create_segment_migration_sankey(df_filtered, df_full)
            st.plotly_chart(fig_sankey, width='stretch', config={'displayModeBar': False})

        with persona_col2:
            st.markdown("#### Demographic Composition")
            # Map display names to column names
            demo_options_map = {
                'Education': 'Education',
                'Gender': 'Gender',
                'Marital Status': 'Marital Status',
                'Income Bracket': 'Income_Bin_Name'
            }
            demo_display = st.selectbox(
                "Select demographic attribute",
                options=list(demo_options_map.keys()),
                key='demo_split'
            )
            demo_attribute = demo_options_map[demo_display]
            fig_demo = create_demographic_split(df_filtered, demo_attribute)
            st.plotly_chart(fig_demo, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== ROW 4: DATA EXPORT ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>Data Export</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        # Define export columns: all filterable columns + cluster assignments
        export_cols = [
            'Loyalty#',
            # FM Segment
            'fm_segment_combined',
            # Cluster assignments
            'Behavioral_Cluster',
            'cluster_name',
            'Demographic_Cluster',
            'demographic_cluster_name',
            # Value-based filters
            'Frequency',
            'Monetary',
            # Behavioral filters
            'redemption_frequency',
            'flight_regularity',
            'companion_flight_ratio',
            'distance_variability',
            # Demographic filters
            'Province or State',
            'City',
            'FSA',
            'Gender',
            'Education',
            'Location Code',
            'Marital Status'
        ]

        # Preview table with renamed columns for display
        st.markdown("##### Data Preview (First 10 Records)")
        preview_df = df_filtered[export_cols].head(10).rename(columns={
            'fm_segment_combined': 'FM Segment',
            'cluster_name': 'Cluster Name',
            'demographic_cluster_name': 'Demographic Cluster Description',
            'redemption_frequency': 'Redemption Frequency',
            'flight_regularity': 'Flight Regularity',
            'companion_flight_ratio': 'Companion Flight Ratio',
            'distance_variability': 'Distance Variability'
        })
        st.dataframe(
            preview_df,
            width='stretch',
            hide_index=True
        )

        # Export button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = df_filtered[export_cols].to_csv(index=False)
            st.download_button(
                label="Download Selected Customers (CSV)",
                data=csv,
                file_name=f"aiai_customers_{len(df_filtered)}.csv",
                mime="text/csv",
                width='stretch',
                type="primary"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #047857; padding: 20px 0;'>
        <p style='margin: 0;'><strong>AIAI Strategic Explorer</strong> | Amazing International Airlines Inc.</p>
        <p style='font-size: 0.8rem; margin-top: 5px; color: #059669;'>Powered by Behavioral Clustering (SOM + K-Means) | PCA Visualization | Group 80 Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
