# Customer Offer Recommendation System

## Overview
This system analyzes customer usage patterns from network data and recommends personalized offers based on a Self-Organizing Map (SOM) machine learning model. The system processes customer behavioral data (like browsing patterns, protocol usage, and data consumption) to group similar customers and suggest appropriate service offers.

## Features
- Customer behavior clustering using Self-Organizing Maps (SOM)
- Bitmap encoding for categorical features (DPI policies, content types, IP protocols)
- Data preprocessing and normalization
- Recommendation storage in SQLite database
- Batch processing for large datasets
- CSV export functionality

## Requirements
- Python 3.6+
- pandas
- numpy
- joblib
- sqlite3
- tqdm
- minisom

## File Structure
```
├── models2/
│   ├── som_model.joblib         # Primary SOM model file
│   ├── cluster_offers.joblib    # Cluster to offer mapping
│   ├── policy_to_bit.joblib     # DPI policy encoding map
│   ├── content_to_bit.joblib    # Content type encoding map
│   ├── proto_to_bit.joblib      # IP protocol encoding map
│   ├── scaler.joblib            # Feature scaling model
│   ├── medians.joblib           # Median values for missing data imputation
│   └── numeric_cols.joblib      # List of numeric columns used in training
├── Dockerfile
├── main_sub_multi.py #API code implementation for batch of customers
├── main_sub1.py #API code implementation for one customer
└── notebook.ipynb #Notebook having all steps of data preparation and modeling
```

## Input Data Format
The system expects a parquet file with the following fields:
- `SubscriberID`: Unique identifier for each customer
- `DpiPolicy`: List of DPI policies used
- `contentType`: List of content types accessed
- `IpProtocol`: List of IP protocols used
- `appName`: List of applications used
- `bytesFromClient`: Upload data volume
- `bytesFromServer`: Download data volume
- `sessions_count`: Number of network sessions
- `transationDuration`: Total duration of sessions

## How It Works
1. **Data Loading**: Loads customer data from parquet file
2. **Preprocessing**: 
   - Converts categorical lists to bitmap encodings
   - Scales numerical features
   - Handles missing values
3. **Recommendation**:
   - Maps customer to nearest SOM node
   - Retrieves pre-assigned offers for that node
   - Falls back to default offers if needed
4. **Storage**:
   - Saves recommendations to SQLite database
   - Optionally exports to CSV

## Key Functions
- `load_models()`: Loads all required ML models and mappings
- `recommend_nbo()`: Core recommendation function using SOM
- `preprocess_customer_data()`: Prepares raw customer data for the model
- `process_all_customers()`: Processes the entire dataset in batches
- `export_to_csv()`: Exports recommendations to CSV format

## Default Recommendations
If a customer has insufficient data or their cluster has no assigned offers, the system falls back to these default offers:
- F3000G100M
- F1200G50M
- F3000G200M

## Performance Considerations
- Uses batch processing to handle large customer datasets
- Commits database transactions periodically to prevent memory issues
- Shows progress bar for long-running operations

## Maintenance
- Models can be retrained as customer behavior evolves
- Cluster-to-offer mappings should be updated when new offers are available