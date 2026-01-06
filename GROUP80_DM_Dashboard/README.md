# AIAI Customer Segmentation Dashboard

An interactive Streamlit dashboard for exploring customer segmentation analysis for Amazing International Airlines Inc. (AIAI). This tool provides visualization and filtering capabilities across behavioral, value-based, and demographic customer segments.

## Live Demo

**Access the dashboard online (no installation required):**

**[https://group80dmproject.streamlit.app](https://group80dmproject.streamlit.app)**

Simply click the link above to start exploring the customer segmentation analysis immediately.

---

## Features

- **3D Customer Landscape**: Interactive PCA-based visualization of customer behavioral clusters
- **FM Matrix Analysis**: Frequency-Monetary value segmentation with quadrant-based insights
- **Segment Migration Flow**: Sankey diagram showing customer distribution across behavioral and value segments
- **Demographic Composition**: Comparative analysis of demographic attributes across segments
- **Advanced Filtering**: Multi-dimensional filtering by behavioral metrics, value segments, and demographics
- **Data Export**: Download filtered customer lists as CSV files

## Quick Start

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

## Detailed Setup (Optional)

### Prerequisites

- Python 3.8+ with pip installed

### What Gets Installed

Running `pip install -r requirements.txt` installs:
- **streamlit** - Dashboard framework
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **plotly** - Interactive visualizations

### Using a Virtual Environment (Recommended for Clean Setup)

If you want to avoid conflicts with other Python projects:

```bash
# Create environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install and run
pip install -r requirements.txt
streamlit run Group80_Cluster_Dashboard.py
```

## Using the Dashboard

### Sidebar Filters

The left sidebar provides comprehensive filtering options:

1. **Focus Group**: Filter by customer loyalty status
   - All customers
   - Active, Current Loyalty Member
   - Active, No Loyalty Member

2. **Value-Based Filters**: Filter by FM segments, frequency, and monetary metrics
   - Value-Based Segments (Champions, Elite, Frequent Flyer, Premium Occasional, At Risk)
   - Frequency Range (flights per active month)
   - Monetary Range (distance per active month)

3. **Behavioral Filters**: Filter by customer behavior patterns
   - Customer Segments (Family Travelers, Business Commuters, Disengaged Solo, Explorers, Engaged Loyalists)
   - Redemption Frequency
   - Flight Regularity
   - Companion Flight Ratio
   - Distance Variability

4. **Demographic Filters**: Filter by customer demographics
   - Province/State
   - City
   - FSA (Forward Sortation Area)
   - Gender
   - Education Level
   - Location Code

### Main Visualizations

1. **3D Customer Landscape**: Rotate and explore the 3D scatter plot to understand customer clustering in behavioral space
2. **FM Matrix**: Hover over points to see individual customer details and FM segment classification
3. **Segment Migration Flow**: Trace how behavioral clusters map to value-based segments
4. **Demographic Composition**: Switch between different demographic attributes to compare segment profiles

### Data Export

- Preview the first 10 records of your filtered selection
- Download the complete filtered dataset as a CSV file
- Exported data includes cluster assignments, behavioral metrics, and demographic attributes

## Data Requirements

The dashboard expects a file named `customer_segmentation_profiles.csv` in the same directory. This file should contain:

- Customer identifiers (Loyalty#)
- Behavioral cluster assignments
- Demographic cluster assignments
- PCA coordinates (pca_1, pca_2, pca_3)
- Behavioral metrics (redemption_frequency, companion_flight_ratio, flight_regularity, distance_variability)
- FM analysis metrics (Frequency, Monetary, fm_segment_combined, fm_tier_combined)
- Demographic attributes (Province, City, Gender, Education, Income, Marital Status, etc.)

## Troubleshooting

### Port Already in Use

If you see an error that port 8501 is already in use, you can:
- Stop the existing Streamlit process
- Run on a different port: `streamlit run Group80_Cluster_Dashboard.py --server.port 8502`

### Module Not Found Errors

If you encounter import errors:
1. Ensure you've installed all requirements: `pip install -r requirements.txt`
2. Verify you're using the correct Python environment
3. Check your Python version is 3.8 or higher: `python --version`

### Data File Not Found

If the dashboard shows a "Data file not found" error:
- Ensure `customer_segmentation_profiles.csv` is in the same directory as the dashboard script
- Check the file name spelling matches exactly

## Project Context

This dashboard is part of Deliverable 2 for the Data Mining course (Group 80), focusing on customer segmentation analysis for Amazing International Airlines Inc. (AIAI). The analysis follows the CRISP-DM framework and provides actionable insights for customer targeting and loyalty program optimization.

## Authors

Group 80 - Data Mining Project Team

## License

This project is developed for academic purposes as part of a university course project.