# NYC Taxi Fare Prediction & Analysis

![Platform](https://img.shields.io/badge/Platform-Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A comprehensive big data analysis and machine learning project built on **Databricks** examining NYC taxi trip records (2009-2024) for both Yellow and Green taxi services. This project leverages the power of Apache Spark on Databricks to analyze nearly 1 billion trip records, uncovering patterns in urban mobility, revenue distribution, and fare prediction at scale.

> **🔥 Built entirely on Databricks Unified Analytics Platform** - Leveraging PySpark, Spark SQL, Unity Catalog, and distributed computing to process 977M+ trip records at scale.

## 📊 Project Overview

This project performs an end-to-end analysis of New York City taxi data on **Databricks**, combining data engineering, statistical analysis, and machine learning to:

- Clean and process ~1 billion taxi trip records using **Apache Spark on Databricks**
- Analyze temporal and spatial patterns in taxi usage with **PySpark and Spark SQL**
- Identify revenue-generating routes and optimal trip durations through **distributed computing**
- Build predictive models for fare estimation using **Databricks ML Runtime**
- Provide actionable insights for drivers and operators leveraging **Databricks Notebooks** for interactive analysis

### Why Databricks?

This project requires processing nearly 1 billion records, making traditional single-machine approaches infeasible. **Databricks provides**:
- **Unified Analytics Platform**: Seamlessly combines data engineering (Spark) and ML workflows
- **Collaborative Notebooks**: Interactive PySpark, SQL, and Python environments
- **Unity Catalog**: Organized data management with catalogs, schemas, and volumes
- **Auto-scaling Clusters**: Distributed processing power for big data operations
- **Built-in Visualizations**: Native support for data exploration and insights

## 🎯 Key Objectives

1. **Data Processing**: Clean and integrate Yellow and Green taxi datasets with location data
2. **Statistical Analysis**: Examine trip characteristics, revenue patterns, and tipping behavior
3. **Urban Mobility Insights**: Identify peak times, popular routes, and borough flow patterns
4. **Predictive Modeling**: Build ML models to predict taxi fares based on trip features
5. **Business Intelligence**: Provide recommendations for maximizing driver income

## 📁 Dataset

### Data Sources
- **Yellow Taxi Data**: NYC TLC Yellow Taxi Trip Records (2009-2024)
- **Green Taxi Data**: NYC TLC Green Taxi Trip Records (2013-2024)
- **Location Data**: NYC Taxi Zone Lookup (Borough and Zone information)

### Dataset Size
- **Total Records**: ~977 million trips (after cleaning)
- **Time Period**: 2009 - 2024
- **Geographic Coverage**: All NYC boroughs (Manhattan, Queens, Brooklyn, Bronx, Staten Island)

### Key Features
- Temporal: `pickup_datetime`, `dropoff_datetime`
- Spatial: `PULocationID`, `DOLocationID`, `pickup_borough`, `dropoff_borough`
- Trip Metrics: `trip_distance`, `duration_sec`, `speed_kph`
- Financial: `fare_amount`, `tip_amount`, `tolls_amount`, `total_amount`
- Categorical: `taxi_color`, `passenger_count`, `payment_type`

## 🛠️ Technical Stack

### Core Platform
- **Databricks Unified Analytics Platform**
  - Databricks Runtime for Apache Spark 3.x
  - Unity Catalog for data governance
  - Databricks Volumes for scalable storage
  - Collaborative Notebooks for interactive development

### Languages & Frameworks
- **PySpark**: Distributed data processing at scale
- **Spark SQL**: High-performance SQL queries on big data
- **Python**: Data science and ML implementation
- **dbutils**: Databricks utilities for file system operations

### ML & Analytics Libraries
- **scikit-learn**: Machine learning models (Random Forest, Linear Regression)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization

### Data Management
- **Parquet Format**: Columnar storage optimized for Spark
- **Delta Lake** (optional): ACID transactions and time travel capabilities
- **Unity Catalog**: Centralized data governance (`workspace.bde.assignment2`)

## 🔧 Setup & Installation

### Prerequisites
- **Databricks Workspace** (Community Edition or Enterprise)
- **Databricks Runtime**: 11.3 LTS or higher (with Spark 3.x)
- **Python**: 3.8+ (pre-installed in Databricks)
- **Cluster Configuration**: 
  - Minimum: 2 workers (8GB RAM each)
  - Recommended: Auto-scaling cluster for large datasets

### Required Libraries (installed via notebook)
```python
# Auto-installed in the notebook
- gdown (for data download from Google Drive)
- scikit-learn (for ML models)
- pandas, numpy (for data processing)
- matplotlib, seaborn (for visualization)
```

### Running the Notebook on Databricks

#### Step 1: Import the Notebook
1. Download or clone this repository
2. Log into your **Databricks workspace**
3. Navigate to **Workspace** → **Import**
4. Upload `25217353_AT_2.ipynb`

#### Step 2: Create a Compute Cluster
1. Go to **Compute** in the sidebar
2. Click **Create Cluster**
3. Configure:
   - **Databricks Runtime**: 11.3 LTS or higher
   - **Cluster Mode**: Standard
   - **Worker type**: Auto-scaling (2-8 workers recommended)

#### Step 3: Configure Unity Catalog (Cells 0-1)
The notebook automatically sets up the data environment:
```python
catalog = "workspace"
schema  = "bde"
volume  = "assignment2"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {catalog}.{schema}.{volume}")
```

#### Step 4: Execute the Analysis
- Attach the notebook to your cluster
- Run all cells sequentially (Runtime: ~30-60 minutes for full dataset)
- The notebook handles data download, cleaning, analysis, and ML modeling automatically

### Data Ingestion on Databricks
The notebook uses **Databricks Volumes** for scalable data storage and `gdown` for fetching data:
```python
# Download to Databricks Volume
GREEN_DST = f"/Volumes/{catalog}/{schema}/{volume}/green"
YELLOW_DST = f"/Volumes/{catalog}/{schema}/{volume}/yellow"

# Create directories using dbutils
dbutils.fs.mkdirs(GREEN_DST)
dbutils.fs.mkdirs(YELLOW_DST)

# Fetch data from Google Drive
fetch_to_volume("1XlToYPB2fJ9Ky6VlNLIjWggDBXvpF-Cd", GREEN_DST, "green_taxi.parquet")
fetch_to_volume("1W-f4gXbYrfqS6nWoLestn_1xBJSq8M1r", YELLOW_DST, "yellow_taxi.parquet")

# Read with Spark for distributed processing
green_df = spark.read.parquet(GREEN_DST)
yellow_df = spark.read.parquet(YELLOW_DST)
```

## 🔥 Databricks-Specific Features Used

This project leverages several **Databricks-native capabilities**:

### 1. **Unity Catalog** 
- Organized data governance with three-level namespace: `catalog.schema.volume`
- Centralized metadata management
- Schema enforcement and data lineage

### 2. **Databricks Volumes**
- Scalable file storage integrated with Unity Catalog
- Direct access via `/Volumes/` path for Parquet files
- Managed storage lifecycle

### 3. **Mixed Language Support**
- **PySpark cells**: For data transformations and cleaning
- **SQL cells** (using `%sql` magic): For complex aggregations and analytics
- **Python cells**: For pandas-based ML modeling

### 4. **Spark SQL Integration**
- Created temporary views: `taxi_trips`, `taxi_trips_cleaned`
- Leveraged Spark SQL for distributed GROUP BY and aggregations
- Used window functions (`RANK()`, `PERCENTILE_CONT()`) at scale

### 5. **dbutils Filesystem API**
- `dbutils.fs.mkdirs()`: Create volume directories
- `dbutils.fs.ls()`: List files in volumes
- Seamless integration with Databricks File System (DBFS)

### 6. **Distributed Processing**
- Processed 977M+ records using Spark's distributed engine
- Used `unionByName()` for efficient DataFrame merging
- Leveraged lazy evaluation and query optimization

### 7. **Interactive Visualizations**
- Native `display()` function for instant DataFrame visualization
- Integrated matplotlib/seaborn for custom plots
- Real-time query results in notebook cells

## 📈 Analysis Components

### Part 1: Exploratory Data Analysis (Q1)

**Key Questions:**
- Which day of the week had the most trips?
- Which hour of the day had the most trips?
- What was the average number of passengers?
- What was the average fare per trip and per passenger?

**Key Findings:**
- Trip volumes peaked in 2013-2014 at over 16M trips/month
- COVID-19 caused a dramatic drop to 263K trips in April 2020
- Recovery to ~3M trips/month by 2023-2024
- Average passenger count remained stable at ~1.6 passengers/trip

### Part 2: Trip Characteristics (Q2)

**Metrics Analyzed:**
- Average, median, min, max trip duration (minutes)
- Average, median, min, max trip distance (km)
- Average, median, min, max speed (km/h)

**Key Findings:**
- **Green Taxis**: 13.8 min avg, 3.05 km avg, 12.8 km/h avg speed
- **Yellow Taxis**: 14.4 min avg, 3.06 km avg, 11.8 km/h avg speed
- Median trip duration: ~11 minutes
- Median trip distance: ~2 km
- Most trips are short, low-speed urban journeys

### Part 3: Borough Flow Analysis (Q3-Q4)

**Analysis Approach:**
This section leverages **Spark SQL on Databricks** for distributed aggregations:
```sql
-- Executed on Databricks using %sql magic command
-- Processes 977M records using distributed GROUP BY
SELECT
  taxi_color, pickup_borough, dropoff_borough,
  YEAR(pickup_datetime) AS year,
  COUNT(*) AS total_trips,
  ROUND(AVG(trip_distance), 2) AS avg_distance_km,
  ROUND(SUM(total_amount), 2) AS total_amount_paid
FROM taxi_trips_cleaned
GROUP BY taxi_color, pickup_borough, dropoff_borough, YEAR(pickup_datetime)
ORDER BY total_trips DESC;
```

**Analysis Breakdown:**
- Total trips by borough pair
- Average distance per route
- Average fare per trip
- Total revenue by pickup→dropoff combinations

**Revenue Insights:**
| Route | Revenue | % of Total |
|-------|---------|-----------|
| Manhattan → Manhattan | $708.6M | 62.6% |
| Queens → Manhattan | $171.7M | 15.2% |
| Manhattan → Queens | $73.8M | 6.5% |
| Queens → Brooklyn | ~3.4% | 3.4% |
| Manhattan → Brooklyn | ~2.9% | 2.9% |

**Top 10 borough pairs account for 97.2% of total revenue**

### Part 4: Tipping Behavior (Q5-Q6)

**Q5: Percentage of trips with tips**
- **62.8%** of trips included tips (613.9M out of 977.3M)

**Q6: High-value tips (≥$15)**
- Only **0.83%** of tipped trips had tips ≥$15 (5.1M out of 613.9M)
- Most tips are modest; generous tips are rare

### Part 5: Duration Analysis (Q7-Q8)

**Trip Duration Bins:**
| Duration | Total Trips | Avg Fare | Fare/Minute |
|----------|-------------|----------|-------------|
| Under 5 mins | High volume | $7 | $2.34 |
| 5-10 mins | Very high | $12 | $1.50 |
| 10-20 mins | **342M** (highest) | **$16.41** | **$1.16** |
| 20-30 mins | Moderate | $23 | $1.15 |
| 30-60 mins | Lower | $47 | ≤$1.25 |
| 60+ mins | Rare | $71 | Low |

**Q8: Optimal Duration for Drivers**
- **Recommendation: Target 10-20 minute trips**
- Reasons:
  - Highest trip volume (342M trips)
  - Balanced fare ($16.41 avg)
  - Sustainable efficiency ($1.16/min)
  - Best for maximizing daily income

## 🤖 Machine Learning Implementation

### Models Evaluated
1. **Baseline Model** (Mean predictor)
2. **Linear Regression**
3. **Random Forest Regressor**

### Feature Engineering
```python
features = [
    "trip_distance", "duration_sec", "tip_amount", "speed_kph",
    "airport_fee", "improvement_surcharge", "extra",
    "pickup_month", "pickup_dow", "pickup_hour",
    "PULocationID", "DOLocationID",
    "fare_efficiency",  # tip_amount / trip_distance
    "duration_per_km"   # duration_sec / trip_distance
]
```

### Train/Test Split
- **Training Data**: All trips before Oct 2024
- **Test Data**: Oct-Dec 2024 (holdout period)
- **Sample Size**: 1% random sample (~9.7M trips) for computational efficiency

### Model Performance (RMSE)

| Model | RMSE | Improvement vs Baseline |
|-------|------|------------------------|
| Baseline (Mean) | 25.9 | - |
| Linear Regression | 10.0 | 61% reduction |
| **Random Forest** | **4.6** | **82% reduction** |

### Best Model: Random Forest
- **Hyperparameters**: 100 estimators, max_depth=12
- **RMSE on Holdout Set**: 4.6
- **Why it works**: Captures non-linear relationships between distance, duration, surcharges, and location features

## 🔍 Data Cleaning Methodology

### Databricks + Spark Approach
Data cleaning on **Databricks** leverages distributed processing to handle 1B+ records efficiently:
- **Lazy Evaluation**: Transformations are optimized before execution
- **Distributed Filtering**: Filters applied across cluster nodes in parallel
- **Column-level Operations**: PySpark's vectorized operations for speed
- **Catalyst Optimizer**: Automatic query optimization by Spark SQL

### Cleaning Filters Applied
```python
def clean_trips_percentile(df):
    # Applied using PySpark DataFrame API
    # Duration: 30 sec - 200 min (12,000 sec)
    # Distance: 0.1 km - 100 km
    # Speed: 0 - 120 km/h (excluding nulls)
    # Passenger count: 1-6 (impute 0/null → 1, cap at 6)
    # Valid datetime range: 2009-2024
    # Remove future dates, negative fares, invalid locations
    
    # Executed in parallel across Spark cluster
    df = df.filter(
        (col("duration_sec") >= 30) & (col("duration_sec") <= 12000)
    )
    # ... additional distributed filters
```

### Data Quality Metrics
- **Records Retained**: ~98.7% of original data
- **Records Removed**: ~1.3% (well within 10% threshold)
- **Removal Reasons**:
  - Invalid/future dates
  - Unrealistic speeds (>120 km/h)
  - Zero or negative fares
  - Missing critical location data
  - Green taxi records before 2013 (service didn't exist)

### Data Integration
- Unified Yellow and Green taxi schemas
- Added `taxi_color` field for differentiation
- Joined with location lookup to add borough information
- Created computed fields: `duration_sec`, `speed_kph`

## 📊 Key Insights & Business Recommendations

### For Taxi Drivers
1. **Target 10-20 minute trips** for optimal income/time ratio
2. **Focus on Manhattan routes** (62%+ of revenue)
3. **Airport runs** (Queens-Manhattan) are highly profitable
4. **Peak hours**: Analyze hourly patterns in your borough

### For Fleet Operators
1. **Manhattan-centric operations** dominate revenue
2. **Queens-Manhattan corridor** is the second-most valuable
3. **Tipping culture is strong** (63% of trips) — encourage quality service
4. **Short trips dominate volume** — optimize for high turnover

### For Urban Planners
1. **Manhattan remains the taxi hub** despite decline since 2015
2. **COVID-19 impact was severe** but recovery is underway
3. **Inter-borough connectivity** is critical (Queens-Manhattan flow)
4. **Average speeds are low** (~12 km/h) — traffic congestion is significant

## ⚡ Performance & Scalability

### Why This Project Requires Databricks

Processing 977 million records is beyond the capability of traditional single-machine approaches:

| Metric | Scale | Databricks Advantage |
|--------|-------|---------------------|
| **Total Records** | 977M+ trips | Distributed across cluster nodes |
| **Data Size** | ~50GB+ Parquet | Parallel I/O with Parquet columnar format |
| **Aggregations** | Billion+ group operations | Spark's Catalyst optimizer + distributed shuffles |
| **Join Operations** | 977M × location lookup | Broadcast joins for efficiency |
| **ML Training** | 9.7M sample (1%) | pandas_udf for distributed feature engineering |

### Processing Time Estimates
On a **Databricks cluster** (4-8 workers, 32GB RAM each):
- **Data ingestion**: ~5-10 minutes
- **Data cleaning**: ~15-20 minutes
- **SQL aggregations**: ~10-15 minutes per complex query
- **ML model training**: ~5-10 minutes
- **Total runtime**: ~45-60 minutes

On a **single machine** (16GB RAM):
- Likely to run out of memory or take 6-12+ hours

### Optimization Techniques Used
1. **Parquet columnar format** for efficient column pruning
2. **Broadcast joins** for small dimension tables (location lookup)
3. **Partition pruning** by filtering on `pickup_datetime` year
4. **Lazy evaluation** to optimize execution plans
5. **Caching** intermediate DataFrames (`taxi_trips_cleaned`)
6. **Column pruning** to read only necessary fields

## 📚 Project Structure

```
nyc-taxi-fare-prediction/
│
├── 25217353_AT_2.ipynb          # Main Databricks notebook
├── README.md                     # This file
├── LICENSE                       # MIT License
│
└── Data (in Databricks Volumes)
    ├── /Volumes/workspace/bde/assignment2/
    │   ├── green/green_taxi.parquet
    │   ├── yellow/yellow_taxi.parquet
    │   └── Taxi_CSV/taxi_zone_lookup.csv
```

## 🎓 Academic Context

**Course**: Big Data Engineering  
**Institution**: University of Technology Sydney (UTS)  
**Student ID**: 25217353  
**Assignment**: AT_2

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Databricks** for providing the unified analytics platform that made processing 1 billion records feasible through:
  - Scalable Apache Spark infrastructure
  - Unity Catalog for data governance
  - Collaborative notebook environment
  - Auto-scaling compute clusters
  - Seamless integration of data engineering and ML workflows
- **NYC Taxi & Limousine Commission (TLC)** for providing comprehensive, open-access taxi trip data
- **Apache Spark** for the powerful distributed computing framework
- **University of Technology Sydney (UTS)** for the Big Data Engineering course that enabled this project

## 📞 Contact

For questions or collaboration:
- GitHub: [@SatyamPalkar](https://github.com/SatyamPalkar)
- Repository: [nyc-taxi-fare-prediction](https://github.com/SatyamPalkar/nyc-taxi-fare-prediction)

---

## ⚠️ Important Notes

**Platform Requirement**: This project is built specifically for **Databricks** and requires:
- Access to a Databricks workspace (Community or Enterprise edition)
- A Spark cluster for distributed processing
- Unity Catalog for data organization
- The notebook cannot run on standard Jupyter without significant modifications due to Databricks-specific features (`dbutils`, Unity Catalog, Spark configuration)

**Data Disclaimer**: This project analyzes historical data for academic purposes. Results should not be used for real-time operational decisions without further validation and testing on current data.
