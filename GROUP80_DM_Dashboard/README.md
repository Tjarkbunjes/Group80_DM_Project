<p>
  <img src="img/Group80_Background.png" alt="Group80 Background" />
</p>

# AIAI Customer Segmentation Dashboard

An interactive Streamlit dashboard for exploring customer segmentation analysis for Amazing International Airlines Inc. (AIAI). This tool provides comprehensive visualization and filtering capabilities across behavioral, value-based, and demographic customer segments.

## Features

- **3D Customer Landscape**: Interactive PCA-based visualization of customer behavioral clusters
- **FM Matrix Analysis**: Frequency-Monetary value segmentation with quadrant-based insights
- **Segment Migration Flow**: Sankey diagram showing customer distribution across behavioral and value segments
- **Demographic Composition**: Comparative analysis of demographic attributes across segments
- **Advanced Filtering**: Multi-dimensional filtering by behavioral metrics, value segments, and demographics
- **Data Export**: Download filtered customer lists as CSV files

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### Step 1: Download the Dashboard

Clone or download this repository to your local machine.

### Step 2: Navigate to the Dashboard Directory

Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and navigate to the dashboard folder:

```bash
cd path/to/GROUP80_DM_Dashboard
```

Replace `path/to/` with the actual path where you downloaded the dashboard.

### Step 3: Install Required Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (v1.52.2)
- pandas (v2.3.3)
- numpy (v2.3.5)
- plotly (v6.5.0)

**Note**: If you're using a virtual environment (recommended), activate it before installing dependencies:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Then install requirements
pip install -r requirements.txt
```

## Running the Dashboard

### Step 4: Launch Streamlit

Once all dependencies are installed, run the dashboard with:

```bash
streamlit run Group80_Cluster_Dashboard.py
```

### Step 5: Access the Dashboard

After running the command, Streamlit will:
1. Start a local web server
2. Automatically open your default web browser
3. Display the dashboard at `http://localhost:8501`

If the browser doesn't open automatically, you can manually navigate to the URL shown in the terminal (typically `http://localhost:8501`).

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