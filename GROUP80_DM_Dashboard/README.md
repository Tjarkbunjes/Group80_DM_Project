# AIAI Customer Segmentation Dashboard

An interactive Streamlit dashboard for exploring customer segmentation analysis for Amazing International Airlines Inc. (AIAI). This tool provides visualization and filtering capabilities across behavioral, value-based, and demographic customer segments.

## Live Demo

**Access the dashboard online (no installation required):**

**[https://group80dmproject.streamlit.app](https://group80dmproject.streamlit.app)**

Simply click the link above to start exploring the customer segmentation analysis immediately.

## Local Setup

### 1. Open Terminal in Dashboard Folder

Navigate to the `GROUP80_DM_Dashboard` folder and open your terminal here.

### 2. Install Dependencies (First Time Only)

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run Group80_Cluster_Dashboard.py
```

That's it! The dashboard will automatically open in your browser at `http://localhost:8501`.

---

## Features

- **3D Customer Landscape**: Interactive PCA-based visualization of customer behavioral clusters
- **FM Matrix Analysis**: Frequency-Monetary value segmentation with quadrant-based insights
- **Segment Migration Flow**: Sankey diagram showing customer distribution across behavioral and value segments
- **Demographic Composition**: Comparative analysis of demographic attributes across segments
- **Advanced Filtering**: Multi-dimensional filtering by behavioral metrics, value segments, and demographics
- **Data Export**: Download filtered customer lists as CSV files

---

## Using the Dashboard

### Sidebar Filters

The left sidebar provides comprehensive filtering options:

1. **Focus Group**: Filter by customer loyalty status

2. **Value-Based Filters**: Filter by FM segments, frequency, and monetary metrics

3. **Behavioral Filters**: Filter by customer behavior patterns

4. **Demographic Filters**: Filter by customer demographics

### Main Visualizations

1. **3D Customer Landscape**: Rotate and explore the 3D scatter plot to understand customer clustering in behavioral space
2. **FM Matrix**: Hover over points to see individual customer details and FM segment classification
3. **Segment Migration Flow**: Trace how behavioral clusters map to value-based segments
4. **Demographic Composition**: Switch between different demographic attributes to compare segment profiles

### Data Export

- Preview the first 10 records of your filtered selection
- Download the complete filtered dataset as a CSV file
- Exported data includes cluster assignments, behavioral metrics, and demographic attributes

---

## Project Context

This dashboard is part of Deliverable 2 Options for the Data Mining course (Group 80), focusing on customer segmentation analysis for Amazing International Airlines Inc. (AIAI).

## Authors

Group 80 - Data Mining Project Team

- 20250487 - PaulHarnos
- 20250489 - Felix Diederichs
- 20250505 - Tjark Bunjes