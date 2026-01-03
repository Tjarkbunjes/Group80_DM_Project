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

# --- Custom CSS with Green Palette ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e1a;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
        border-right: 2px solid #10b981;
    }

    /* Make multiselect and slider widgets green */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #10b981 !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #10b981 !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: #10b981 !important;
    }

    /* Headers */
    h1 {
        color: #10b981 !important;
        font-weight: 700 !important;
    }

    h2, h3, h4 {
        color: #d1fae5 !important;
    }

    /* Metrics cards */
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #6ee7b7 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    [data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #065f46 0%, #047857 100%);
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 20px 0 10px 0;
    }

    /* Filter section */
    .filter-box {
        background-color: #1a1f2e;
        border: 1px solid #065f46;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }

    /* Data source footer */
    .data-source {
        background-color: #065f46;
        color: #d1fae5;
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

# --- Behavioral Cluster Configuration (Group 80 Green Palette) ---
CLUSTER_CONFIG = {
    0: {
        'name': 'Disengaged Solo',
        'color': '#6b7280',  # Gray - Low engagement
        'desc': 'Low engagement, infrequent redemption, solo travelers.'
    },
    1: {
        'name': 'Business Commuters',
        'color': '#3b82f6',  # Blue - Regular but transactional
        'desc': 'Regular solo business travelers, moderate engagement.'
    },
    2: {
        'name': 'Engaged Loyalists',
        'color': '#10b981',  # Primary Green - High value
        'desc': 'Active, consistent, high redemption, loyal base.'
    },
    3: {
        'name': 'Family Travelers',
        'color': '#f59e0b',  # Amber - Companion-based
        'desc': 'High companion ratio, vacation-oriented patterns.'
    },
    4: {
        'name': 'Explorers',
        'color': '#8b5cf6',  # Purple - Variable behavior
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

    # Add cluster names
    df['cluster_name'] = df['Behavioral_Cluster'].map(lambda x: CLUSTER_CONFIG.get(x, {}).get('name', f'Cluster {x}'))

    return df

# --- Visualization Functions ---

def create_3d_universe(df: pd.DataFrame) -> go.Figure:
    """Create the 3D scatter plot - The Universe visualization."""

    fig = go.Figure()

    # Store cluster counts for annotation
    cluster_counts = {}

    for cluster_id in sorted(df['Behavioral_Cluster'].unique()):
        cluster_df = df[df['Behavioral_Cluster'] == cluster_id]
        cluster_info = CLUSTER_CONFIG.get(cluster_id, {})
        cluster_name = cluster_info.get('name', f'Cluster {cluster_id}')
        cluster_color = cluster_info.get('color', '#6b7280')

        # Store count
        cluster_counts[cluster_name] = len(cluster_df)

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
                  f"━━━━━━━━━━━━<br>"
                  f"<b>Demographics:</b><br>"
                  f"• Income: ${row['Income']:,.0f}<br>"
                  f"• Education: {row['Education']}<br>"
                  f"• Province: {row['Province or State']}"
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
                title=dict(text="<b>PC1: Engagement Axis</b>", font=dict(color='#10b981', size=12)),
                backgroundcolor='#0a0e1a',
                gridcolor='#065f46',
                showbackground=True,
                range=axis_range
            ),
            yaxis=dict(
                title=dict(text="<b>PC2: Travel Pattern Axis</b>", font=dict(color='#10b981', size=12)),
                backgroundcolor='#0a0e1a',
                gridcolor='#065f46',
                showbackground=True,
                range=axis_range
            ),
            zaxis=dict(
                title=dict(text="<b>PC3: Secondary Pattern Axis</b>", font=dict(color='#10b981', size=12)),
                backgroundcolor='#0a0e1a',
                gridcolor='#065f46',
                showbackground=True,
                range=axis_range
            ),
            bgcolor='#0a0e1a',
            aspectmode='cube'
        ),
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0a0e1a',
        font=dict(color='#d1fae5'),
        legend=dict(
            bgcolor='rgba(6, 95, 70, 0.8)',
            bordercolor='#10b981',
            borderwidth=2,
            font=dict(color='#d1fae5', size=11),
            title=dict(text="<b>Customer Segments</b>", font=dict(size=12, color='#10b981')),
            x=1.0,
            y=0.9
        ),
        margin=dict(l=0, r=0, t=0, b=100),
        height=600,
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    )

    # Add cluster count annotation box below legend
    count_text = "<b>Cluster Counts:</b><br>"
    for cluster_name in sorted(cluster_counts.keys()):
        count_text += f"• {cluster_name}: {cluster_counts[cluster_name]:,}<br>"
    count_text += f"<br><b>Total: {len(df):,}</b>"

    fig.add_annotation(
        text=count_text,
        xref="paper", yref="paper",
        x=1.0, y=-0.1,
        xanchor="right", yanchor="top",
        showarrow=False,
        bgcolor='rgba(6, 95, 70, 0.8)',
        bordercolor='#10b981',
        borderwidth=2,
        borderpad=10,
        font=dict(color='#d1fae5', size=10),
        align='left'
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
            bgcolor='#1a1f2e',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#065f46',
                tickfont=dict(color='#6ee7b7', size=9),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '25', '50', '75', '100']
            ),
            angularaxis=dict(
                gridcolor='#065f46',
                tickfont=dict(color='#d1fae5', size=10)
            )
        ),
        paper_bgcolor='#0a0e1a',
        font=dict(color='#d1fae5'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(6, 95, 70, 0.6)',
            bordercolor='#10b981',
            borderwidth=1,
            font=dict(color='#d1fae5')
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

    colors_demo = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5']

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
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#d1fae5'),
        xaxis=dict(
            title=dict(text=f'<b>Distribution (%)</b>', font=dict(color='#10b981')),
            gridcolor='#065f46',
            tickfont=dict(color='#6ee7b7'),
            range=[0, 100]
        ),
        yaxis=dict(
            title=dict(text='<b>Customer Segment</b>', font=dict(color='#10b981')),
            tickfont=dict(color='#d1fae5'),
            showgrid=False
        ),
        legend=dict(
            bgcolor='rgba(6, 95, 70, 0.6)',
            bordercolor='#10b981',
            borderwidth=1,
            font=dict(color='#d1fae5'),
            title=dict(text=f'<b>{attribute}</b>', font=dict(color='#10b981'))
        ),
        margin=dict(l=10, r=10, t=20, b=40),
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
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#1a1f2e',
            font=dict(color='#d1fae5')
        )
        return fig

    # Filter to only rows with FM segment data
    df_with_fm = df[df[fg_column].notna()].copy()

    # Calculate medians and 90th percentiles
    freq_median = df_with_fm['Frequency'].median()
    mon_median = df_with_fm['Monetary'].median()
    freq_p90 = df_with_fm['Frequency'].quantile(0.90)
    mon_p90 = df_with_fm['Monetary'].quantile(0.90)

    # Define colors
    segment_colors = {
        'Champions': '#10b981',      # Green
        'Frequent Flyer': '#3b82f6', # Blue
        'Premium Occasional': '#f59e0b', # Amber
        'At Risk': '#ec4899'         # Pink
    }
    elite_color = '#8b5cf6'  # Purple for Elite

    fig = go.Figure()

    # Add quadrant backgrounds
    max_freq = df_with_fm['Frequency'].max() * 1.05
    max_mon = df_with_fm['Monetary'].max() * 1.05

    # Champions quadrant highlight
    fig.add_shape(type="rect",
        x0=freq_median, x1=max_freq, y0=mon_median, y1=max_mon,
        fillcolor='rgba(16, 185, 129, 0.08)', line=dict(width=0), layer='below')

    # Elite zone highlight
    fig.add_shape(type="rect",
        x0=freq_p90, x1=max_freq, y0=mon_p90, y1=max_mon,
        fillcolor='rgba(139, 92, 246, 0.15)', line=dict(width=0), layer='below')

    # Add median reference lines
    fig.add_hline(y=mon_median, line_dash="dash", line_color="#6b7280", line_width=2, opacity=0.7)
    fig.add_vline(x=freq_median, line_dash="dash", line_color="#6b7280", line_width=2, opacity=0.7)

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

    # Add quadrant labels as annotations
    fig.add_annotation(x=freq_median * 1.4, y=mon_median * 1.4,
        text="Champions", showarrow=False,
        font=dict(size=14, color='#d1fae5', family="Arial Black"),
        bgcolor='rgba(10, 14, 26, 0.7)', borderpad=4)

    fig.add_annotation(x=freq_median * 0.3, y=mon_median * 1.4,
        text="Premium<br>Occasional", showarrow=False,
        font=dict(size=14, color='#d1fae5', family="Arial Black"),
        bgcolor='rgba(10, 14, 26, 0.7)', borderpad=4)

    fig.add_annotation(x=freq_median * 1.4, y=mon_median * 0.3,
        text="Frequent<br>Flyer", showarrow=False,
        font=dict(size=14, color='#d1fae5', family="Arial Black"),
        bgcolor='rgba(10, 14, 26, 0.7)', borderpad=4)

    fig.add_annotation(x=freq_median * 0.3, y=mon_median * 0.3,
        text="At Risk", showarrow=False,
        font=dict(size=14, color='#d1fae5', family="Arial Black"),
        bgcolor='rgba(10, 14, 26, 0.7)', borderpad=4)

    # Add Elite label
    elite_center_x = (freq_p90 + max_freq) / 2
    elite_center_y = (mon_p90 + max_mon) / 2
    fig.add_annotation(x=elite_center_x, y=elite_center_y,
        text="Elite", showarrow=False,
        font=dict(size=13, color='#d1fae5', family="Arial Black"),
        bgcolor='rgba(139, 92, 246, 0.3)', borderpad=4)

    fig.update_layout(
        title=dict(
            text=f'FM Matrix: Value-Based Segmentation<br><sub>n={len(df_with_fm):,} customers</sub>',
            font=dict(color='#10b981', size=14)
        ),
        xaxis=dict(
            title=dict(text='<b>Frequency (Flights per Active Month)</b>', font=dict(color='#10b981')),
            gridcolor='#065f46',
            tickfont=dict(color='#6ee7b7'),
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='<b>Monetary (Distance per Active Month)</b>', font=dict(color='#10b981')),
            gridcolor='#065f46',
            tickfont=dict(color='#6ee7b7'),
            showgrid=True,
            zeroline=False
        ),
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#d1fae5'),
        legend=dict(
            bgcolor='rgba(6, 95, 70, 0.6)',
            bordercolor='#10b981',
            borderwidth=1,
            font=dict(color='#d1fae5', size=9)
        ),
        hovermode='closest',
        height=500
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

        # --- BEHAVIORAL FILTERS ---
        with st.expander("Behavioral Filters", expanded=True):
            # Select All button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Reset All", key="reset_behavioral"):
                    st.rerun()

            # Customer Segments
            segment_options = [CLUSTER_CONFIG[i]['name'] for i in sorted(CLUSTER_CONFIG.keys())]
            selected_segments = st.multiselect(
                "Customer Segments",
                options=segment_options,
                default=segment_options
            )

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

            # Average Distance per Flight
            avg_distance_range = st.slider(
                "Avg Distance per Flight",
                min_value=float(df_full['avg_distance_per_flight'].min()),
                max_value=float(df_full['avg_distance_per_flight'].max()),
                value=(float(df_full['avg_distance_per_flight'].min()), float(df_full['avg_distance_per_flight'].max())),
                step=10.0
            )

            # Peak Season Sin
            peak_season_range = st.slider(
                "Peak Season (Sin)",
                min_value=float(df_full['peak_season_sin'].min()),
                max_value=float(df_full['peak_season_sin'].max()),
                value=(float(df_full['peak_season_sin'].min()), float(df_full['peak_season_sin'].max())),
                step=0.01
            )

        # --- DEMOGRAPHIC FILTERS ---
        with st.expander("Demographic Filters", expanded=False):
            # Reset button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Reset All", key="reset_demographic"):
                    st.rerun()

            # Province
            province_options = sorted([p for p in df_full['Province or State'].unique() if pd.notna(p)])
            selected_provinces = st.multiselect(
                "Province",
                options=province_options,
                default=province_options
            )

            # City
            city_options = sorted([c for c in df_full['City'].unique() if pd.notna(c) and c != 'Unknown'])
            selected_cities = st.multiselect(
                "City",
                options=city_options,
                default=city_options
            )

            # FSA
            fsa_options = sorted([f for f in df_full['FSA'].unique() if pd.notna(f)])
            selected_fsa = st.multiselect(
                "FSA (Forward Sortation Area)",
                options=fsa_options,
                default=fsa_options
            )

            # Gender
            gender_options = sorted([g for g in df_full['Gender'].unique() if pd.notna(g)])
            selected_gender = st.multiselect(
                "Gender",
                options=gender_options,
                default=gender_options
            )

            # Education Level
            education_options = sorted([e for e in df_full['Education'].unique() if pd.notna(e) and e != 'Unknown'])
            selected_education = st.multiselect(
                "Education Level",
                options=education_options,
                default=education_options
            )

            # Location Code
            location_options = sorted([l for l in df_full['Location Code'].unique() if pd.notna(l)])
            selected_location = st.multiselect(
                "Location Code",
                options=location_options,
                default=location_options
            )

        # --- VALUE-BASED FILTERS ---
        with st.expander("Value-Based Filters", expanded=False):
            # Reset button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Reset All", key="reset_value"):
                    st.rerun()

            # Focus Group Selection
            focus_group_options = []
            if 'fm_segment_fg1' in df_full.columns:
                focus_group_options.append('Focus Group 1: Loyalty Members | Active')
            if 'fm_segment_fg2' in df_full.columns:
                focus_group_options.append('Focus Group 2: Ex-Loyalty Members | Active')

            selected_focus_groups = st.multiselect(
                "Focus Groups",
                options=focus_group_options,
                default=focus_group_options
            )

            # Income Range
            income_range = st.slider(
                "Income Range",
                min_value=0,
                max_value=int(df_full['Income'].max()),
                value=(0, int(df_full['Income'].max())),
                format="$%d"
            )

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
        (df_filtered['distance_variability'] <= distance_var_range[1]) &
        (df_filtered['avg_distance_per_flight'] >= avg_distance_range[0]) &
        (df_filtered['avg_distance_per_flight'] <= avg_distance_range[1]) &
        (df_filtered['peak_season_sin'] >= peak_season_range[0]) &
        (df_filtered['peak_season_sin'] <= peak_season_range[1])
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
        (df_filtered['Income'] >= income_range[0]) &
        (df_filtered['Income'] <= income_range[1]) &
        (df_filtered['Frequency'] >= freq_range[0]) &
        (df_filtered['Frequency'] <= freq_range[1]) &
        (df_filtered['Monetary'] >= monetary_range[0]) &
        (df_filtered['Monetary'] <= monetary_range[1])
    ]

    # Apply focus group filter
    if selected_focus_groups:
        focus_group_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        if 'Focus Group 1: Loyalty Members | Active' in selected_focus_groups:
            focus_group_mask |= df_filtered['fm_segment_fg1'].notna()
        if 'Focus Group 2: Ex-Loyalty Members | Active' in selected_focus_groups:
            focus_group_mask |= df_filtered['fm_segment_fg2'].notna()
        df_filtered = df_filtered[focus_group_mask]

    # ==================== MAIN HEADER ====================
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2.5rem; margin: 0;'>AIAI Customer Segmentation Strategy</h1>
        <p style='color: #6ee7b7; font-size: 1.1rem; margin-top: 10px;'>Behavioral Clustering & Value Analysis | Group 80</p>
    </div>
    """, unsafe_allow_html=True)

    # Top-Level KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        delta_pct = (len(df_filtered) / len(df_full) * 100) - 100
        st.metric(
            "Selected Customers",
            f"{len(df_filtered):,}",
            delta=f"{delta_pct:.1f}% of base" if delta_pct < 0 else f"{len(df_filtered)/len(df_full)*100:.1f}% of base"
        )

    with kpi_col2:
        avg_redemption = df_filtered['redemption_frequency'].mean()
        pop_redemption = df_full['redemption_frequency'].mean()
        delta_redemption = ((avg_redemption - pop_redemption) / pop_redemption * 100) if pop_redemption > 0 else 0
        st.metric(
            "Avg. Redemption Rate",
            f"{avg_redemption:.1%}",
            delta=f"{delta_redemption:+.1f}% vs pop"
        )

    with kpi_col3:
        avg_distance = df_filtered['avg_distance_per_flight'].mean()
        pop_distance = df_full['avg_distance_per_flight'].mean()
        delta_distance = ((avg_distance - pop_distance) / pop_distance * 100) if pop_distance > 0 else 0
        st.metric(
            "Avg. Distance Flown",
            f"{avg_distance:,.0f} km",
            delta=f"{delta_distance:+.1f}% vs pop"
        )

    with kpi_col4:
        avg_clv = df_filtered['Customer Lifetime Value'].mean()
        st.metric(
            "Avg. CLV",
            f"${avg_clv:,.0f}"
        )

    st.markdown("---")

    # ==================== ROW 1: 3D VISUALIZATION ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>3D Customer Landscape</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        fig_3d = create_3d_universe(df_filtered)
        st.plotly_chart(fig_3d, use_container_width=True, config={'displaylogo': False})
    else:
        st.warning("No customers match the current filters. Please adjust your selection.")

    st.markdown("---")

    # ==================== ROW 2: FM MATRIX SEGMENTATION ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>FM Matrix: Value-Based Segmentation</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        fig_fm = create_fm_matrix_combined(df_filtered)
        st.plotly_chart(fig_fm, use_container_width=True, config={'displaylogo': False})
    else:
        st.warning("No customers match the current filters. Please adjust your selection.")

    st.markdown("---")

    # ==================== ROW 3: COMPARATIVE ANALYSIS ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>Comparative Analysis: Selected vs Population</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        persona_col1, persona_col2 = st.columns(2)

        with persona_col1:
            st.markdown("#### Behavioral Signature")
            fig_radar = create_persona_radar(df_filtered, df_full)
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

        with persona_col2:
            st.markdown("#### Demographic Composition")
            demo_attribute = st.selectbox(
                "Select demographic attribute",
                options=['Education', 'Gender', 'Marital Status'],
                key='demo_split'
            )
            fig_demo = create_demographic_split(df_filtered, demo_attribute)
            st.plotly_chart(fig_demo, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")

    # ==================== ROW 4: DATA EXPORT ====================
    st.markdown("<div class='section-header'><h3 style='margin: 0;'>Data Export</h3></div>", unsafe_allow_html=True)

    if len(df_filtered) > 0:
        # Preview table
        st.markdown("##### Data Preview (First 10 Records)")
        preview_cols = ['Loyalty#', 'cluster_name', 'Education', 'Income', 'Province or State',
                       'redemption_frequency', 'Frequency', 'Monetary', 'Customer Lifetime Value']
        st.dataframe(
            df_filtered[preview_cols].head(10),
            use_container_width=True,
            hide_index=True
        )

        # Export button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download Selected Customers (CSV)",
                data=csv,
                file_name=f"aiai_customers_{len(df_filtered)}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6ee7b7; padding: 20px 0;'>
        <p style='margin: 0;'><strong>AIAI Strategic Explorer</strong> | Amazing International Airlines Inc.</p>
        <p style='font-size: 0.8rem; margin-top: 5px; color: #10b981;'>Powered by Behavioral Clustering (SOM + K-Means) | PCA Visualization | Group 80 Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
