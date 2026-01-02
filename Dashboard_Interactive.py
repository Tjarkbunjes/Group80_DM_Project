"""
Interactive Cluster Visualization Dashboard - Deliverable 2
AIAI Customer Segmentation Dashboard with Advanced Filtering

Features:
- Three-column layout (Filter Panel | 3D Visualization | Deep-Dive Analytics)
- Demographic and value-based filtering
- Interactive 3D PCA cluster visualization
- Comprehensive cluster analytics and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="AIAI Interactive Segmentation Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }

    h1, h2, h3, h4 {
        color: #c9d1d9 !important;
    }

    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.3rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.85rem !important;
    }

    .filter-section {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .cluster-card {
        background-color: #21262d;
        border-left: 4px solid;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Behavioral Cluster Configuration (for 3D PCA visualization) ---
BEHAVIORAL_CLUSTER_CONFIG = {
    0: {
        'name': 'Behavioral Cluster 0',
        'color': '#ef4444',  # Red
        'desc': 'Low engagement behavioral pattern.'
    },
    1: {
        'name': 'Behavioral Cluster 1',
        'color': '#3b82f6',  # Blue
        'desc': 'Moderate behavioral engagement.'
    },
    2: {
        'name': 'Behavioral Cluster 2',
        'color': '#10b981',  # Green
        'desc': 'Active behavioral pattern.'
    },
    3: {
        'name': 'Behavioral Cluster 3',
        'color': '#f59e0b',  # Orange
        'desc': 'High companion-based behavior.'
    },
    4: {
        'name': 'Behavioral Cluster 4',
        'color': '#a855f7',  # Purple (brighter)
        'desc': 'Premium behavioral pattern.'
    }
}

# --- FM Segment Configuration (for filtering and analysis) ---
FM_SEGMENT_CONFIG = {
    'Champions': {
        'name': 'Champions',
        'color': '#10b981',  # Emerald/Green
        'desc': 'High-value frequent flyers with strong engagement and loyalty.'
    },
    'Frequent Flyer': {
        'name': 'Frequent Flyer',
        'color': '#3b82f6',  # Blue
        'desc': 'Regular travelers with consistent flight patterns and moderate spending.'
    },
    'At Risk': {
        'name': 'At Risk',
        'color': '#ec4899',  # Pink
        'desc': 'Customers showing declining engagement, requiring retention efforts.'
    },
    'Premium Occasional': {
        'name': 'Premium Occasional',
        'color': '#f59e0b',  # Amber
        'desc': 'Infrequent but high-value travelers with premium preferences.'
    }
}

def get_behavioral_cluster_info(cluster_id, key='name'):
    """Get behavioral cluster information safely."""
    return BEHAVIORAL_CLUSTER_CONFIG.get(cluster_id, {}).get(key, f'Cluster {cluster_id}' if key == 'name' else '#9ca3af')

def get_fm_segment_info(segment, key='name'):
    """Get FM segment information safely."""
    return FM_SEGMENT_CONFIG.get(segment, {}).get(key, str(segment) if key == 'name' else '#9ca3af')


# --- Data Loading ---
@st.cache_data
def load_data():
    """Load customer segmentation data."""
    df = pd.read_csv('data/clustering_data/customer_segmentation_profiles.csv')

    # Handle missing values in demographics
    df['Income'] = df['Income'].fillna(0)
    df['Education'] = df['Education'].fillna('Unknown')
    df['City'] = df['City'].fillna('Unknown')
    df['Province or State'] = df['Province or State'].fillna('Unknown')
    df['FSA'] = df['FSA'].fillna('Unknown')

    # Handle missing values in FM segments
    df['fm_segment_fg1'] = df['fm_segment_fg1'].fillna('')
    df['fm_segment_fg2'] = df['fm_segment_fg2'].fillna('')

    return df


# --- Visualization Functions ---

def create_3d_scatter(df: pd.DataFrame) -> go.Figure:
    """Create 3D PCA scatter plot with behavioral cluster coloring."""

    fig = go.Figure()

    for cluster_id in sorted(df['Behavioral_Cluster'].unique()):
        cluster_df = df[df['Behavioral_Cluster'] == cluster_id]
        cluster_name = get_behavioral_cluster_info(cluster_id, 'name')
        cluster_color = get_behavioral_cluster_info(cluster_id, 'color')

        fig.add_trace(go.Scatter3d(
            x=cluster_df['pca_1'],
            y=cluster_df['pca_2'],
            z=cluster_df['pca_3'],
            mode='markers',
            name=f"{cluster_name} ({len(cluster_df):,})",
            marker=dict(
                size=4,
                color=cluster_color,
                opacity=0.7
            ),
            text=[f"<b>Customer {row['Loyalty#']}</b><br>"
                  f"Behavioral: {cluster_name}<br>"
                  f"FM Segment: {row['fm_segment_fg1'] if pd.notna(row['fm_segment_fg1']) else row['fm_segment_fg2']}<br>"
                  f"─────────────<br>"
                  f"Frequency: {row['Frequency']:.2f}<br>"
                  f"Monetary: ${row['Monetary']:.2f}<br>"
                  f"Education: {row['Education']}<br>"
                  f"Income: ${row['Income']:,.0f}<br>"
                  f"City: {row['City']}"
                  for _, row in cluster_df.iterrows()],
            hoverinfo='text',
            showlegend=True
        ))

    # Calculate axis ranges to make them equal
    all_points = pd.concat([df['pca_1'], df['pca_2'], df['pca_3']])
    axis_min = all_points.min()
    axis_max = all_points.max()
    axis_range = [axis_min, axis_max]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="PC1 (Behavioral Dimension)", font=dict(color='#c9d1d9', size=11)),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                tickfont=dict(color='#8b949e', size=9),
                range=axis_range
            ),
            yaxis=dict(
                title=dict(text="PC2 (Value Dimension)", font=dict(color='#c9d1d9', size=11)),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                tickfont=dict(color='#8b949e', size=9),
                range=axis_range
            ),
            zaxis=dict(
                title=dict(text="PC3 (Demographic Dimension)", font=dict(color='#c9d1d9', size=11)),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                tickfont=dict(color='#8b949e', size=9),
                range=axis_range
            ),
            bgcolor='#0d1117',
            aspectmode='cube'
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.95)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9', size=10),
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        scene_camera=dict(
            eye=dict(x=1.4, y=1.4, z=1.2)
        )
    )

    return fig


def create_cluster_distribution(df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of FM segment sizes."""

    cluster_counts = df.groupby('fm_segment_fg1').size().reset_index(name='count')
    cluster_counts['segment_name'] = cluster_counts['fm_segment_fg1'].apply(lambda x: get_fm_segment_info(x, 'name'))
    cluster_counts['percentage'] = (cluster_counts['count'] / len(df) * 100)
    cluster_counts = cluster_counts.sort_values('count', ascending=True)

    colors = [get_fm_segment_info(c, 'color') for c in cluster_counts['fm_segment_fg1']]

    fig = go.Figure(go.Bar(
        y=cluster_counts['segment_name'],
        x=cluster_counts['count'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{cnt:,} ({pct:.1f}%)" for cnt, pct in zip(cluster_counts['count'], cluster_counts['percentage'])],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=10)
    ))

    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(
            title=dict(text='Number of Customers', font=dict(color='#8b949e', size=10)),
            gridcolor='#30363d',
            tickfont=dict(color='#8b949e', size=9)
        ),
        yaxis=dict(
            tickfont=dict(color='#c9d1d9', size=10),
            showgrid=False
        ),
        margin=dict(l=10, r=80, t=20, b=40),
        height=250,
        showlegend=False
    )

    return fig


def create_feature_boxplot(df: pd.DataFrame, feature: str, feature_name: str) -> go.Figure:
    """Create box plot for feature distribution across FM segments."""

    fig = go.Figure()

    for segment in sorted(df['fm_segment_fg1'].dropna().unique()):
        segment_data = df[df['fm_segment_fg1'] == segment][feature]
        segment_name = get_fm_segment_info(segment, 'name')
        segment_color = get_fm_segment_info(segment, 'color')

        fig.add_trace(go.Box(
            y=segment_data,
            name=segment_name,
            marker_color=segment_color,
            boxmean=True
        ))

    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        yaxis=dict(
            title=dict(text=feature_name, font=dict(color='#8b949e', size=10)),
            gridcolor='#30363d',
            tickfont=dict(color='#8b949e', size=9)
        ),
        xaxis=dict(
            tickfont=dict(color='#c9d1d9', size=9),
            showgrid=False
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=30),
        height=220
    )

    return fig


def create_cluster_profile_radar(df: pd.DataFrame, segment: str) -> go.Figure:
    """Create radar chart for a specific FM segment's profile."""

    cluster_data = df[df['fm_segment_fg1'] == segment]

    # Calculate average values for key features
    features = {
        'Frequency': cluster_data['Frequency'].mean(),
        'Monetary': cluster_data['Monetary'].mean() / 1000,  # Scale down
        'Dist. Var': cluster_data['distance_variability'].mean(),
        'Companion': cluster_data['companion_flight_ratio'].mean(),
        'Regularity': cluster_data['flight_regularity'].mean(),
        'Redemption': cluster_data['redemption_frequency'].mean()
    }

    # Normalize to 0-1 for radar chart
    all_data_stats = {
        'Frequency': (df['Frequency'].min(), df['Frequency'].max()),
        'Monetary': (df['Monetary'].min() / 1000, df['Monetary'].max() / 1000),
        'Dist. Var': (df['distance_variability'].min(), df['distance_variability'].max()),
        'Companion': (df['companion_flight_ratio'].min(), df['companion_flight_ratio'].max()),
        'Regularity': (df['flight_regularity'].min(), df['flight_regularity'].max()),
        'Redemption': (df['redemption_frequency'].min(), df['redemption_frequency'].max())
    }

    normalized_values = []
    for feat, val in features.items():
        min_val, max_val = all_data_stats[feat]
        if max_val > min_val:
            norm = (val - min_val) / (max_val - min_val)
        else:
            norm = 0.5
        normalized_values.append(max(0, min(1, norm)))

    categories = list(features.keys())
    normalized_values.append(normalized_values[0])  # Close the polygon
    categories.append(categories[0])

    segment_color = get_fm_segment_info(segment, 'color')
    segment_name = get_fm_segment_info(segment, 'name')

    # Convert hex to rgba
    hex_color = segment_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)',
        line=dict(color=segment_color, width=2),
        name=segment_name
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#30363d',
                tickfont=dict(color='#8b949e', size=8),
                tickvals=[0, 0.5, 1],
                ticktext=['Low', 'Med', 'High']
            ),
            angularaxis=dict(
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9', size=9)
            )
        ),
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )

    return fig


def create_income_education_distribution(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of education by income bins."""

    # Create income bins
    df_temp = df.copy()
    df_temp['Income_Bracket'] = pd.cut(
        df_temp['Income'],
        bins=[0, 30000, 60000, 90000, float('inf')],
        labels=['<$30K', '$30-60K', '$60-90K', '$90K+']
    )

    # Count by education and income
    edu_income = df_temp.groupby(['Education', 'Income_Bracket']).size().reset_index(name='count')

    fig = go.Figure()

    education_levels = ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor']
    colors_edu = ['#6366f1', '#8b5cf6', '#a855f7', '#c084fc', '#e9d5ff']

    for edu, color in zip(education_levels, colors_edu):
        edu_data = edu_income[edu_income['Education'] == edu]
        if len(edu_data) > 0:
            fig.add_trace(go.Bar(
                x=edu_data['Income_Bracket'],
                y=edu_data['count'],
                name=edu,
                marker_color=color
            ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(
            title=dict(text='Income Bracket', font=dict(color='#8b949e', size=10)),
            tickfont=dict(color='#c9d1d9', size=9),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text='Count', font=dict(color='#8b949e', size=10)),
            gridcolor='#30363d',
            tickfont=dict(color='#8b949e', size=9)
        ),
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.9)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9', size=9),
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=10, r=10, t=20, b=80),
        height=250
    )

    return fig


# --- Main App ---

def main():
    # Load data
    try:
        df_full = load_data()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please ensure 'data/clustering_data/customer_segmentation_profiles.csv' exists.")
        return

    # Header
    st.markdown("""
    <div style='text-align: center; padding: 15px 0;'>
        <h1 style='color: #58a6ff; margin: 0;'>AIAI Customer Segmentation Dashboard</h1>
        <p style='color: #8b949e; font-size: 0.95rem; margin-top: 5px;'>Interactive Cluster Visualization - Deliverable 2</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Create three columns
    filter_col, viz_col, deepdive_col = st.columns([1.5, 5, 3.5])

    # ==================== LEFT: FILTER PANEL ====================
    with filter_col:
        st.markdown("### Filters")

        # Demographic Filters Section
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("**Demographics**")

            # Education filter
            education_options = ['All'] + sorted(df_full['Education'].unique().tolist())
            selected_education = st.selectbox(
                "Education",
                options=education_options,
                index=0,
                key='education_filter'
            )

            # Income filter
            income_min = float(df_full['Income'].min())
            income_max = float(df_full['Income'].max())
            income_range = st.slider(
                "Income Range",
                min_value=income_min,
                max_value=income_max,
                value=(income_min, income_max),
                format="$%.0f",
                key='income_filter'
            )

            # City filter
            city_options = ['All'] + sorted([c for c in df_full['City'].unique() if c != 'Unknown'])
            selected_city = st.selectbox(
                "City",
                options=city_options,
                index=0,
                key='city_filter'
            )

            # FSA filter
            fsa_options = ['All'] + sorted([f for f in df_full['FSA'].unique() if f != 'Unknown'])
            selected_fsa = st.selectbox(
                "FSA (Postal)",
                options=fsa_options,
                index=0,
                key='fsa_filter'
            )

            # Province filter
            province_options = ['All'] + sorted(df_full['Province or State'].unique().tolist())
            selected_province = st.selectbox(
                "Province",
                options=province_options,
                index=0,
                key='province_filter'
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # Value-Based Filters Section
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("**Value Metrics**")

            # Monetary filter
            monetary_min = float(df_full['Monetary'].min())
            monetary_max = float(df_full['Monetary'].max())
            monetary_range = st.slider(
                "Monetary Value",
                min_value=monetary_min,
                max_value=monetary_max,
                value=(monetary_min, monetary_max),
                format="$%.0f",
                key='monetary_filter'
            )

            # Frequency filter
            frequency_min = float(df_full['Frequency'].min())
            frequency_max = float(df_full['Frequency'].max())
            frequency_range = st.slider(
                "Frequency (Flights)",
                min_value=frequency_min,
                max_value=frequency_max,
                value=(frequency_min, frequency_max),
                format="%.1f",
                key='frequency_filter'
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # FM Segment Selection
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("**FM Segments (FG1)**")

            selected_segments = []
            # Get unique segments from fg1
            unique_segments = sorted([s for s in df_full['fm_segment_fg1'].dropna().unique() if s != ''])
            for segment in unique_segments:
                segment_name = get_fm_segment_info(segment, 'name')
                segment_color = get_fm_segment_info(segment, 'color')

                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.checkbox(
                        segment_name,
                        value=True,
                        key=f'segment_{segment}'
                    ):
                        selected_segments.append(segment)
                with col2:
                    st.markdown(
                        f"<span style='display: inline-block; width: 12px; height: 12px; "
                        f"background-color: {segment_color}; border-radius: 50%;'></span>",
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        # Reset button
        if st.button("Reset Filters", use_container_width=True):
            st.rerun()

    # ==================== APPLY FILTERS ====================
    df_filtered = df_full.copy()

    # Apply demographic filters
    if selected_education != 'All':
        df_filtered = df_filtered[df_filtered['Education'] == selected_education]

    df_filtered = df_filtered[
        (df_filtered['Income'] >= income_range[0]) &
        (df_filtered['Income'] <= income_range[1])
    ]

    if selected_city != 'All':
        df_filtered = df_filtered[df_filtered['City'] == selected_city]

    if selected_fsa != 'All':
        df_filtered = df_filtered[df_filtered['FSA'] == selected_fsa]

    if selected_province != 'All':
        df_filtered = df_filtered[df_filtered['Province or State'] == selected_province]

    # Apply value-based filters
    df_filtered = df_filtered[
        (df_filtered['Monetary'] >= monetary_range[0]) &
        (df_filtered['Monetary'] <= monetary_range[1]) &
        (df_filtered['Frequency'] >= frequency_range[0]) &
        (df_filtered['Frequency'] <= frequency_range[1])
    ]

    # Apply FM segment filter
    if selected_segments:
        df_filtered = df_filtered[df_filtered['fm_segment_fg1'].isin(selected_segments)]

    # ==================== MIDDLE: 3D VISUALIZATION ====================
    with viz_col:
        st.markdown("### 3D Cluster Visualization")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df_full):,}")
        with col2:
            st.metric("Filtered", f"{len(df_filtered):,}")
        with col3:
            pct = (len(df_filtered) / len(df_full) * 100) if len(df_full) > 0 else 0
            st.metric("Selection", f"{pct:.1f}%")

        st.markdown(
            "<p style='color: #8b949e; font-size: 0.8rem; text-align: center;'>"
            "Drag to rotate • Scroll to zoom • Hover for details</p>",
            unsafe_allow_html=True
        )

        if len(df_filtered) > 0:
            fig_3d = create_3d_scatter(df_filtered)
            st.plotly_chart(fig_3d, use_container_width=True, config={'displaylogo': False})
        else:
            st.warning("No customers match the current filters. Please adjust your selection.")

        # Export button
        if len(df_filtered) > 0:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Export Filtered Data (CSV)",
                data=csv,
                file_name="aiai_filtered_customers.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ==================== RIGHT: DEEP DIVE ====================
    with deepdive_col:
        st.markdown("### Deep Dive Analytics")

        if len(df_filtered) > 0:
            # Cluster Distribution
            st.markdown("#### Cluster Distribution")
            fig_dist = create_cluster_distribution(df_filtered)
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

            # Segment Selector for Profile
            st.markdown("#### FM Segment Profile")
            segment_for_profile = st.selectbox(
                "Select FM segment to analyze",
                options=sorted([s for s in df_filtered['fm_segment_fg1'].dropna().unique() if s != '']),
                format_func=lambda x: get_fm_segment_info(x, 'name'),
                key='profile_segment'
            )

            if segment_for_profile is not None:
                # Segment description
                segment_desc = get_fm_segment_info(segment_for_profile, 'desc')
                segment_color = get_fm_segment_info(segment_for_profile, 'color')
                segment_name = get_fm_segment_info(segment_for_profile, 'name')

                st.markdown(
                    f"<div class='cluster-card' style='border-left-color: {segment_color};'>"
                    f"<h4 style='margin: 0; color: {segment_color};'>{segment_name}</h4>"
                    f"<p style='color: #8b949e; font-size: 0.85rem; margin: 5px 0 0 0;'>{segment_desc}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Radar chart
                fig_radar = create_cluster_profile_radar(df_filtered, segment_for_profile)
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

                # Segment statistics
                segment_subset = df_filtered[df_filtered['fm_segment_fg1'] == segment_for_profile]

                st.markdown("**Key Statistics**")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Customers", f"{len(segment_subset):,}")
                    st.metric("Avg Frequency", f"{segment_subset['Frequency'].mean():.1f}")
                    st.metric("Avg CLV", f"${segment_subset['Customer Lifetime Value'].mean():.0f}")
                with stat_col2:
                    st.metric("% of Total", f"{len(segment_subset)/len(df_filtered)*100:.1f}%")
                    st.metric("Avg Monetary", f"${segment_subset['Monetary'].mean():.0f}")
                    st.metric("Avg Distance", f"{segment_subset['avg_distance_per_flight'].mean():.0f} km")

            st.markdown("---")

            # Feature Distribution Analysis
            st.markdown("#### Feature Analysis")
            feature_option = st.selectbox(
                "Select feature to analyze",
                options=[
                    ('Frequency', 'Flight Frequency'),
                    ('Monetary', 'Monetary Value'),
                    ('distance_variability', 'Distance Variability'),
                    ('companion_flight_ratio', 'Companion Ratio'),
                    ('flight_regularity', 'Flight Regularity'),
                    ('redemption_frequency', 'Redemption Frequency')
                ],
                format_func=lambda x: x[1],
                key='feature_analysis'
            )

            if feature_option:
                fig_box = create_feature_boxplot(df_filtered, feature_option[0], feature_option[1])
                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

            # Demographics
            st.markdown("#### Demographics Overview")
            fig_demo = create_income_education_distribution(df_filtered)
            st.plotly_chart(fig_demo, use_container_width=True, config={'displayModeBar': False})

        else:
            st.info("Apply filters to view analytics")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8b949e; padding: 10px 0;'>
        <p style='margin: 0; font-size: 0.85rem;'>AIAI Customer Segmentation Dashboard | Group 80</p>
        <p style='font-size: 0.75rem; margin-top: 3px;'>Behavioral Clustering with Multi-dimensional Analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
