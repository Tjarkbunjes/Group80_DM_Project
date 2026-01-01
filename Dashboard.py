"""
AeroAnalytics - AIAI Loyalty Segmentation Dashboard
Streamlit Dashboard für die echten AIAI Clustering-Ergebnisse

Features:
- Interaktive 3D PCA-Visualisierung der Cluster
- Echtzeit-Filterung nach Features und Clustern
- Cluster-Analysen: Radar-Charts, Populationsgrößen, Feature-Verteilungen
- Kunden-Detail-Ansicht
- Daten-Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="AeroAnalytics - AIAI Segmentation",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for dark theme styling ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #c9d1d9 !important;
    }
    
    /* Metrics cards */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    /* Custom info boxes */
    .info-box {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Cluster Configuration ---
# Basierend auf den Cluster-Profilen:
# Cluster 0: Hohe companion_ratio, hohe regularity, hohe redemption → "Social Redeemers"
# Cluster 1: Hohe distance_variability, solo travelers, niedrige redemption → "Solo Explorers"  
# Cluster 2: Niedrige Werte überall → "Occasional Travelers"

CLUSTER_CONFIG = {
    0: {
        'name': 'Social Redeemers',
        'color': '#10b981',  # Emerald/Green
        'desc': 'Frequent travelers with companions, high point redemption, regular flight patterns.'
    },
    1: {
        'name': 'Solo Explorers',
        'color': '#3b82f6',  # Blue
        'desc': 'Variable distance travelers, fly alone, low point redemption, flexible schedules.'
    },
    2: {
        'name': 'Occasional Travelers',
        'color': '#ec4899',  # Pink
        'desc': 'Infrequent flyers, low engagement, stable short-distance patterns.'
    },
}

# Feature descriptions for tooltips
FEATURE_INFO = {
    'distance_variability': 'Variability in flight distances (standardized)',
    'companion_flight_ratio': 'Ratio of flights taken with companions (standardized)',
    'flight_regularity': 'Regularity/consistency of flight patterns (standardized)',
    'redemption_frequency': 'Frequency of loyalty point redemptions (standardized)'
}

def get_cluster_color(cluster_id: int) -> str:
    """Get the color for a cluster."""
    return CLUSTER_CONFIG.get(cluster_id, {}).get('color', '#9ca3af')

def get_cluster_name(cluster_id: int) -> str:
    """Get the display name for a cluster."""
    return CLUSTER_CONFIG.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load the AIAI clustering data from CSV files."""

    
    # Lade Hauptdaten
    main_data = pd.read_csv('data/clustering_data/dashboard_main_data.csv')
    
    # Lade Cluster-Statistiken
    cluster_stats = pd.read_csv('data/clustering_data/cluster_statistics.csv')
    
    # Lade Cluster-Profile
    cluster_profiles = pd.read_csv('data/clustering_data/cluster_profiles.csv')
    
    # Lade PCA-Metadaten
    pca_metadata = pd.read_csv('data/clustering_data/pca_metadata.csv')
    
    # Füge Cluster-Namen hinzu
    main_data['cluster_name'] = main_data['Cluster_SOM_KMeans'].map(
        lambda x: get_cluster_name(x)
    )
    
    return main_data, cluster_stats, cluster_profiles, pca_metadata


# --- Visualization Functions ---

def create_3d_scatter(df: pd.DataFrame, pca_metadata: pd.DataFrame) -> go.Figure:
    """Create an interactive 3D PCA scatter plot of customer clusters."""
    
    fig = go.Figure()
    
    # Add scatter points for each cluster
    for cluster_id in sorted(df['Cluster_SOM_KMeans'].unique()):
        cluster_df = df[df['Cluster_SOM_KMeans'] == cluster_id]
        cluster_name = get_cluster_name(cluster_id)
        cluster_color = get_cluster_color(cluster_id)
        
        fig.add_trace(go.Scatter3d(
            x=cluster_df['PC1'],
            y=cluster_df['PC2'],
            z=cluster_df['PC3'],
            mode='markers',
            name=f"{cluster_name} ({len(cluster_df):,})",
            marker=dict(
                size=4,
                color=cluster_color,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=[f"<b>Customer {row['Loyalty#']}</b><br>"
                  f"Cluster: {cluster_name}<br>"
                  f"─────────────<br>"
                  f"Distance Var: {row['distance_variability']:.2f}<br>"
                  f"Companion Ratio: {row['companion_flight_ratio']:.2f}<br>"
                  f"Flight Regularity: {row['flight_regularity']:.2f}<br>"
                  f"Redemption Freq: {row['redemption_frequency']:.2f}"
                  for _, row in cluster_df.iterrows()],
            hoverinfo='text'
        ))
    
    # Get variance explained for axis labels
    var_explained = pca_metadata.set_index('component')['variance_explained'].to_dict()
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=f"PC1 ({var_explained.get('PC1', 0)*100:.1f}% var)",
                    font=dict(color='#c9d1d9')
                ),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                zerolinecolor='#30363d',
                tickfont=dict(color='#8b949e')
            ),
            yaxis=dict(
                title=dict(
                    text=f"PC2 ({var_explained.get('PC2', 0)*100:.1f}% var)",
                    font=dict(color='#c9d1d9')
                ),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                zerolinecolor='#30363d',
                tickfont=dict(color='#8b949e')
            ),
            zaxis=dict(
                title=dict(
                    text=f"PC3 ({var_explained.get('PC3', 0)*100:.1f}% var)",
                    font=dict(color='#c9d1d9')
                ),
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                showbackground=True,
                zerolinecolor='#30363d',
                tickfont=dict(color='#8b949e')
            ),
            bgcolor='#0d1117'
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.9)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9'),
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600
    )
    
    # Set initial camera position
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    )
    
    return fig


def create_radar_chart(cluster_profiles: pd.DataFrame, selected_clusters: list) -> go.Figure:
    """Create a radar chart showing cluster profiles."""
    
    features = ['distance_variability', 'companion_flight_ratio', 
                'flight_regularity', 'redemption_frequency']
    feature_labels = ['Distance\nVariability', 'Companion\nRatio', 
                      'Flight\nRegularity', 'Redemption\nFrequency']
    
    fig = go.Figure()
    
    for cluster_id in selected_clusters:
        cluster_data = cluster_profiles[cluster_profiles['Cluster_SOM_KMeans'] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        # Get values and normalize to 0-1 range for visualization
        # Since data is standardized (mean=0, std=1), we'll shift and scale
        values = []
        for feat in features:
            val = cluster_data[feat].values[0]
            # Map from approximately [-2, 2] to [0, 1]
            normalized = (val + 2) / 4
            values.append(max(0, min(1, normalized)))
        
        # Close the polygon
        values.append(values[0])
        labels = feature_labels + [feature_labels[0]]
        
        cluster_name = get_cluster_name(cluster_id)
        cluster_color = get_cluster_color(cluster_id)
        
        # Convert hex to rgba for fill
        hex_color = cluster_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            name=cluster_name,
            fill='toself',
            fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)',
            line=dict(color=cluster_color, width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#30363d',
                tickfont=dict(color='#8b949e', size=8),
                linecolor='#30363d',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Low', '', 'Avg', '', 'High']
            ),
            angularaxis=dict(
                gridcolor='#30363d',
                tickfont=dict(color='#8b949e', size=9),
                linecolor='#30363d'
            )
        ),
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.8)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9', size=10),
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=40, b=80),
        height=380
    )
    
    return fig


def create_population_chart(cluster_stats: pd.DataFrame, selected_clusters: list) -> go.Figure:
    """Create a horizontal bar chart showing cluster population sizes."""
    
    filtered_stats = cluster_stats[cluster_stats['cluster_id'].isin(selected_clusters)].copy()
    filtered_stats['cluster_name'] = filtered_stats['cluster_id'].map(get_cluster_name)
    filtered_stats = filtered_stats.sort_values('size', ascending=True)
    
    colors = [get_cluster_color(c) for c in filtered_stats['cluster_id']]
    
    fig = go.Figure(go.Bar(
        x=filtered_stats['size'],
        y=filtered_stats['cluster_name'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:,} ({p:.1f}%)" for s, p in zip(filtered_stats['size'], filtered_stats['percentage'])],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=11)
    ))
    
    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(
            title=dict(
                text='Number of Customers',
                font=dict(color='#8b949e')
            ),
            gridcolor='#30363d',
            tickfont=dict(color='#8b949e')
        ),
        yaxis=dict(
            tickfont=dict(color='#c9d1d9'),
            showgrid=False
        ),
        margin=dict(l=10, r=80, t=20, b=40),
        height=200
    )
    
    return fig


def create_feature_distribution(df: pd.DataFrame, feature: str, selected_clusters: list) -> go.Figure:
    """Create a box plot showing feature distribution per cluster."""
    
    filtered_df = df[df['Cluster_SOM_KMeans'].isin(selected_clusters)].copy()
    filtered_df['cluster_name'] = filtered_df['Cluster_SOM_KMeans'].map(get_cluster_name)
    
    fig = go.Figure()
    
    for cluster_id in selected_clusters:
        cluster_data = filtered_df[filtered_df['Cluster_SOM_KMeans'] == cluster_id][feature]
        cluster_name = get_cluster_name(cluster_id)
        
        fig.add_trace(go.Box(
            y=cluster_data,
            name=cluster_name,
            marker_color=get_cluster_color(cluster_id),
            boxmean=True
        ))
    
    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        yaxis=dict(
            title=dict(
                text=feature.replace('_', ' ').title(),
                font=dict(color='#8b949e')
            ),
            gridcolor='#30363d',
            tickfont=dict(color='#8b949e'),
            zeroline=True,
            zerolinecolor='#58a6ff',
            zerolinewidth=1
        ),
        xaxis=dict(
            tickfont=dict(color='#c9d1d9'),
            showgrid=False
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=40),
        height=250
    )
    
    return fig


def create_histogram_with_range(df: pd.DataFrame, column: str, range_values: tuple) -> go.Figure:
    """Create a histogram with highlighted range."""
    
    fig = go.Figure()
    
    # Full histogram (dimmed)
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=25,
        marker_color='#30363d',
        opacity=0.5,
        name='All Data'
    ))
    
    # Highlighted range
    filtered = df[(df[column] >= range_values[0]) & (df[column] <= range_values[1])]
    fig.add_trace(go.Histogram(
        x=filtered[column],
        nbinsx=25,
        marker_color='#3b82f6',
        opacity=0.8,
        name='Selected'
    ))
    
    # Add range indicators
    fig.add_vline(x=range_values[0], line=dict(color='#60a5fa', dash='dash', width=2))
    fig.add_vline(x=range_values[1], line=dict(color='#60a5fa', dash='dash', width=2))
    
    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e', size=9)),
        yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e', size=9)),
        showlegend=False,
        barmode='overlay',
        margin=dict(l=10, r=10, t=10, b=30),
        height=100
    )
    
    return fig


# --- Main App ---

def main():
    # Load data
    try:
        main_data, cluster_stats, cluster_profiles, pca_metadata = load_data()
    except FileNotFoundError as e:
        st.error(f"""
        **Datei nicht gefunden!**
        
        Bitte stelle sicher, dass alle CSV-Dateien im gleichen Verzeichnis wie das Script liegen:
        - `dashboard_main_data.csv`
        - `cluster_statistics.csv`
        - `cluster_profiles.csv`
        - `pca_metadata.csv`
        
        Fehler: {e}
        """)
        return
    
    df = main_data.copy()
    
    # --- Sidebar: Filters ---
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #58a6ff; margin: 0;'>✈️ AeroAnalytics</h1>
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 5px;'>AIAI Customer Segmentation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- Cluster Selection ---
        st.markdown("### 🎯 Cluster Selection")
        
        selected_clusters = []
        for cluster_id, config in CLUSTER_CONFIG.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.checkbox(
                    config['name'],
                    value=True,
                    key=f"cluster_{cluster_id}",
                    help=config['desc']
                ):
                    selected_clusters.append(cluster_id)
            with col2:
                st.markdown(
                    f"<span style='display: inline-block; width: 12px; height: 12px; "
                    f"background-color: {config['color']}; border-radius: 50%;'></span>",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # --- Feature Filters ---
        st.markdown("### 🎚️ Feature Filters")
        
        # Distance Variability
        st.markdown("**Distance Variability**")
        dist_min, dist_max = float(df['distance_variability'].min()), float(df['distance_variability'].max())
        dist_range = st.slider(
            "dist_var", min_value=dist_min, max_value=dist_max,
            value=(dist_min, dist_max), format="%.2f",
            label_visibility="collapsed"
        )
        st.plotly_chart(
            create_histogram_with_range(df, 'distance_variability', dist_range),
            use_container_width=True, config={'displayModeBar': False}
        )
        
        # Companion Flight Ratio
        st.markdown("**Companion Flight Ratio**")
        comp_min, comp_max = float(df['companion_flight_ratio'].min()), float(df['companion_flight_ratio'].max())
        comp_range = st.slider(
            "comp_ratio", min_value=comp_min, max_value=comp_max,
            value=(comp_min, comp_max), format="%.2f",
            label_visibility="collapsed"
        )
        st.plotly_chart(
            create_histogram_with_range(df, 'companion_flight_ratio', comp_range),
            use_container_width=True, config={'displayModeBar': False}
        )
        
        # Flight Regularity
        st.markdown("**Flight Regularity**")
        reg_min, reg_max = float(df['flight_regularity'].min()), float(df['flight_regularity'].max())
        reg_range = st.slider(
            "flight_reg", min_value=reg_min, max_value=reg_max,
            value=(reg_min, reg_max), format="%.2f",
            label_visibility="collapsed"
        )
        st.plotly_chart(
            create_histogram_with_range(df, 'flight_regularity', reg_range),
            use_container_width=True, config={'displayModeBar': False}
        )
        
        # Redemption Frequency
        st.markdown("**Redemption Frequency**")
        red_min, red_max = float(df['redemption_frequency'].min()), float(df['redemption_frequency'].max())
        red_range = st.slider(
            "redeem_freq", min_value=red_min, max_value=red_max,
            value=(red_min, red_max), format="%.2f",
            label_visibility="collapsed"
        )
        st.plotly_chart(
            create_histogram_with_range(df, 'redemption_frequency', red_range),
            use_container_width=True, config={'displayModeBar': False}
        )
        
        st.markdown("---")
        
        # Export Button
        if st.button("📥 Export Filtered Data", use_container_width=True, type="primary"):
            st.session_state.show_export = True
    
    # --- Apply Filters ---
    filtered_df = df[
        (df['distance_variability'] >= dist_range[0]) & (df['distance_variability'] <= dist_range[1]) &
        (df['companion_flight_ratio'] >= comp_range[0]) & (df['companion_flight_ratio'] <= comp_range[1]) &
        (df['flight_regularity'] >= reg_range[0]) & (df['flight_regularity'] <= reg_range[1]) &
        (df['redemption_frequency'] >= red_range[0]) & (df['redemption_frequency'] <= red_range[1]) &
        (df['Cluster_SOM_KMeans'].isin(selected_clusters if selected_clusters else [-1]))
    ]
    
    # --- Main Content ---
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    filtered_customers = len(filtered_df)
    
    with col1:
        st.metric(
            label="👥 Total Customers",
            value=f"{total_customers:,}"
        )
    
    with col2:
        st.metric(
            label="🔍 Filtered Selection",
            value=f"{filtered_customers:,}",
            delta=f"{filtered_customers/total_customers*100:.1f}%" if total_customers > 0 else "0%"
        )
    
    with col3:
        clusters_shown = len(selected_clusters)
        st.metric(
            label="🎯 Clusters Shown",
            value=f"{clusters_shown} / 3"
        )
    
    with col4:
        total_var = pca_metadata['variance_explained'].sum() * 100
        st.metric(
            label="📊 PCA Variance",
            value=f"{total_var:.1f}%"
        )
    
    st.markdown("---")
    
    # Export download (if triggered)
    if st.session_state.get('show_export', False):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv,
            file_name="aiai_filtered_segments.csv",
            mime="text/csv"
        )
        st.session_state.show_export = False
    
    # Main layout: 3D plot + Analytics
    main_col, analytics_col = st.columns([2, 1])
    
    with main_col:
        st.markdown("### 🌐 3D PCA Cluster Visualization")
        st.markdown(
            "<p style='color: #8b949e; font-size: 0.85rem;'>"
            "Drag to rotate • Scroll to zoom • Hover for customer details</p>",
            unsafe_allow_html=True
        )
        
        if len(filtered_df) > 0:
            fig_3d = create_3d_scatter(filtered_df, pca_metadata)
            st.plotly_chart(fig_3d, use_container_width=True, config={'displaylogo': False})
        else:
            st.warning("Keine Daten entsprechen den aktuellen Filtern. Bitte Auswahl anpassen.")
        
        # Customer Detail Section
        st.markdown("### 🔍 Customer Detail Lookup")
        
        if len(filtered_df) > 0:
            # Customer selector
            customer_options = filtered_df['Loyalty#'].astype(str).tolist()
            selected_customer_id = st.selectbox(
                "Select a customer ID to view details",
                options=[''] + customer_options,
                format_func=lambda x: "Choose a customer..." if x == '' else f"Customer {x}"
            )
            
            if selected_customer_id and selected_customer_id != '':
                customer = filtered_df[filtered_df['Loyalty#'] == int(selected_customer_id)].iloc[0]
                cluster_id = customer['Cluster_SOM_KMeans']
                cluster_name = get_cluster_name(cluster_id)
                cluster_color = get_cluster_color(cluster_id)
                
                detail_cols = st.columns([1, 1, 1])
                
                with detail_cols[0]:
                    st.markdown(f"""
                    <div style='background: #21262d; padding: 15px; border-radius: 8px; border-left: 4px solid {cluster_color};'>
                        <h4 style='margin: 0; color: #c9d1d9;'>Customer {customer['Loyalty#']}</h4>
                        <span style='background: {cluster_color}40; color: {cluster_color}; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;'>{cluster_name}</span>
                        <p style='color: #8b949e; font-style: italic; margin-top: 10px; font-size: 0.85rem;'>{CLUSTER_CONFIG[cluster_id]['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with detail_cols[1]:
                    st.markdown(f"""
                    <div style='background: #21262d; padding: 15px; border-radius: 8px;'>
                        <h5 style='color: #8b949e; margin-top: 0;'>Behavioral Features</h5>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: #8b949e;'>📏 Distance Var</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['distance_variability']:.3f}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: #8b949e;'>👥 Companion Ratio</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['companion_flight_ratio']:.3f}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color: #8b949e;'>📅 Flight Regularity</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['flight_regularity']:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with detail_cols[2]:
                    st.markdown(f"""
                    <div style='background: #21262d; padding: 15px; border-radius: 8px;'>
                        <h5 style='color: #8b949e; margin-top: 0;'>Position & Points</h5>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: #8b949e;'>🎁 Redemption Freq</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['redemption_frequency']:.3f}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: #8b949e;'>📍 PC1</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['PC1']:.3f}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color: #8b949e;'>📍 PC2 / PC3</span>
                            <span style='color: #c9d1d9; font-weight: bold;'>{customer['PC2']:.2f} / {customer['PC3']:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with analytics_col:
        st.markdown("### 📊 Cluster Analytics")
        
        # Radar Chart - Cluster Profiles
        st.markdown("#### Cluster Profiles")
        st.markdown(
            "<p style='color: #8b949e; font-size: 0.75rem;'>"
            "Normalized feature averages per cluster</p>",
            unsafe_allow_html=True
        )
        
        if selected_clusters:
            fig_radar = create_radar_chart(cluster_profiles, selected_clusters)
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Select clusters to view profiles")
        
        # Population Size
        st.markdown("#### Population Distribution")
        if selected_clusters:
            fig_pop = create_population_chart(cluster_stats, selected_clusters)
            st.plotly_chart(fig_pop, use_container_width=True, config={'displayModeBar': False})
        
        # Feature Distribution Selector
        st.markdown("#### Feature Distribution")
        feature_to_show = st.selectbox(
            "Select feature",
            options=['distance_variability', 'companion_flight_ratio', 
                     'flight_regularity', 'redemption_frequency'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_clusters and len(filtered_df) > 0:
            fig_dist = create_feature_distribution(filtered_df, feature_to_show, selected_clusters)
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
        
        # Cluster Statistics Table
        st.markdown("#### Cluster Statistics")
        if selected_clusters:
            stats_data = []
            for cluster_id in selected_clusters:
                cluster_data = cluster_stats[cluster_stats['cluster_id'] == cluster_id]
                if len(cluster_data) > 0:
                    row = cluster_data.iloc[0]
                    stats_data.append({
                        'Cluster': get_cluster_name(cluster_id),
                        'Size': f"{int(row['size']):,}",
                        '%': f"{row['percentage']:.1f}%",
                        'Dist Var': f"{row['mean_distance_variability']:.2f}",
                        'Comp Ratio': f"{row['mean_companion_flight_ratio']:.2f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8b949e; padding: 20px 0;'>
        <p style='margin: 0;'>AeroAnalytics Dashboard | Amazing International Airlines Inc.</p>
        <p style='font-size: 0.8rem; margin-top: 5px;'>SOM + K-Means Customer Segmentation | PCA Visualization</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()