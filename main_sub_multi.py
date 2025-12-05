import pandas as pd
import numpy as np
import joblib
import sqlite3
from tqdm import tqdm
import ast
import json
import os

# === Load saved models ===
def load_models():
    global som_weights, cluster_offers, policy_to_bit, content_to_bit, proto_to_bit, scaler, medians, numeric_cols
    
    # Load SOM model and extract weights
    try:
        # Option 1: Load full SOM model and extract weights
        som_model = joblib.load("models2/som_model.joblib")
        som_weights = som_model._weights  # Extract weights from the SOM model
        print("Successfully loaded SOM weights from model")
    except (FileNotFoundError, AttributeError) as e:
        # Fallback to direct numpy file if som_model.joblib doesn't exist or doesn't have weights attribute
        try:
            som_weights = np.load("som_weights1.npy")
            print("Loaded SOM weights from numpy file")
        except FileNotFoundError:
            raise RuntimeError("Neither SOM model nor weights file could be found")
    
    try:
        cluster_offers = joblib.load('models2/cluster_offers.joblib')
        print(f"Loaded {len(cluster_offers)} cluster offer mappings")
    except FileNotFoundError:
        print("Warning: cluster_offers.joblib not found. Using default fallback.")
        # Fallback to hardcoded values if file not found
        cluster_offers = {i: [] for i in range(100)}  # Initialize with empty lists
    
    # Load encoding mappings
    policy_to_bit = joblib.load('models2/policy_to_bit.joblib')
    content_to_bit = joblib.load('models2/content_to_bit.joblib')
    proto_to_bit = joblib.load('models2/proto_to_bit.joblib')
    
    # Load scaler
    scaler = joblib.load('models2/scaler.joblib')
    
    # Load medians for imputation
    medians = joblib.load('models2/medians.joblib')
    
    # Load numeric columns
    numeric_cols = joblib.load('models2/numeric_cols.joblib')
    
    print("All models loaded successfully!")

# === Define the recommendation function ===

def recommend_nbo(customer_features, som_weights, cluster_offers, som_dims=(10, 10)):
    
    expected_features = som_weights.shape[2]
    if len(customer_features) != expected_features:
        print(f"WARNING: Feature count mismatch. Got {len(customer_features)}, expected {expected_features}")
        if len(customer_features) > expected_features:
            customer_features = customer_features[:expected_features]
        else:
            temp = np.zeros(expected_features)
            temp[:len(customer_features)] = customer_features
            customer_features = temp
    
    customer_features_reshaped = customer_features.reshape(1, 1, -1)
    
    distances = np.linalg.norm(som_weights - customer_features_reshaped, axis=2)
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
    bmu_row, bmu_col = bmu_index
    cluster_id = bmu_row * som_dims[1] + bmu_col
    recommended_offers = cluster_offers.get(cluster_id, [])
    if not recommended_offers:
        return ['F3000G100M', 'F1200G50M', 'F3000G200M']
    return recommended_offers

# === Preprocess customer data function ===
def preprocess_customer_data(customer_data):
    # Create a dictionary to convert to DataFrame
    data_dict = {}
    
    # 1. Encode DpiPolicy to bitmap
    dpi_policies = customer_data.get('DpiPolicy', [])
    if isinstance(dpi_policies, str):
        try:
            dpi_policies = ast.literal_eval(dpi_policies)
        except (ValueError, SyntaxError):
            dpi_policies = []
    
    if dpi_policies:
        bitsum = sum(1 << policy_to_bit.get(p, 0) for p in dpi_policies if p in policy_to_bit)
        data_dict['DpiPolicy'] = bitsum
    else:
        data_dict['DpiPolicy'] = 0
    
    # 2. Encode contentType to bitmap
    content_types = customer_data.get('contentType', [])
    if isinstance(content_types, str):
        try:
            content_types = ast.literal_eval(content_types)
        except (ValueError, SyntaxError):
            content_types = []
    
    if content_types:
        bitsum = sum(1 << content_to_bit.get(c, 0) for c in content_types if c in content_to_bit)
        data_dict['contentType'] = bitsum
    else:
        data_dict['contentType'] = 0
    
    # 3. Encode IpProtocol to bitmap
    protocols = customer_data.get('IpProtocol', [])
    if isinstance(protocols, str):
        try:
            protocols = ast.literal_eval(protocols)
        except (ValueError, SyntaxError):
            protocols = []
    
    if protocols:
        bitsum = sum(1 << proto_to_bit.get(p, 0) for p in protocols if p in proto_to_bit)
        data_dict['IpProtocol'] = bitsum
    else:
        data_dict['IpProtocol'] = 0
    
    # 4. Convert appName to app_count
    app_names = customer_data.get('appName', [])
    if isinstance(app_names, str):
        try:
            app_names = ast.literal_eval(app_names)
        except (ValueError, SyntaxError):
            app_names = []
    
    data_dict['app_count'] = len(app_names) if app_names else 0
    
    # 5. Add other numeric features with fallback to medians if needed
    data_dict['bytesFromClient'] = max(customer_data.get('bytesFromClient', 0), medians.get('bytesFromClient', 1))
    data_dict['bytesFromServer'] = max(customer_data.get('bytesFromServer', 0), medians.get('bytesFromServer', 1))
    data_dict['sessions_count'] = customer_data.get('sessions_count', 0)
    data_dict['transationDuration'] = max(customer_data.get('transationDuration', 0), medians.get('transationDuration', 1))
    
    # Create DataFrame with the processed data
    df = pd.DataFrame([data_dict])
    
    # Ensure all necessary columns are present
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only the columns used in training
    df = df[numeric_cols]
    
    # Apply scaling
    scaled_data = scaler.transform(df)
    
    return scaled_data[0]  # Return first row as numpy array

# === Setup SQLite database ===

def setup_database(db_name="recommendations.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_recommendations (
            SubscriberID TEXT PRIMARY KEY,
            recommended_offers TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

# === Process all customers and save to database ===
def process_all_customers(df, batch_size=1000):
    # Load models
    load_models()
    
    # Setup database
    conn = setup_database()
    cursor = conn.cursor()
    
    # Get total number of unique subscribers
    unique_subscribers = df['SubscriberID'].unique()
    total_subscribers = len(unique_subscribers)
    print(f"Processing recommendations for {total_subscribers} unique subscribers")
    
    # Initialize counter for progress reporting
    processed = 0
    success_count = 0
    error_count = 0
    
    for i in range(0, total_subscribers, batch_size):
        batch_subscribers = unique_subscribers[i:i+batch_size]
        for subscriber_id in tqdm(batch_subscribers, desc=f"Batch {i//batch_size + 1}"):
            try:
                subscriber_data = df[df['SubscriberID'] == subscriber_id].iloc[0].to_dict()
                is_empty = (
                    (not subscriber_data.get('DpiPolicy') or subscriber_data.get('DpiPolicy') == '[]') and
                    (not subscriber_data.get('contentType') or subscriber_data.get('contentType') == '[]') and
                    (not subscriber_data.get('IpProtocol') or subscriber_data.get('IpProtocol') == '[]') and
                    (not subscriber_data.get('appName') or subscriber_data.get('appName') == '[]') and
                    (subscriber_data.get('bytesFromClient', 0) == 0.0) and
                    (subscriber_data.get('bytesFromServer', 0) == 0.0) and
                    (subscriber_data.get('sessions_count', 0) == 0) and
                    (subscriber_data.get('transationDuration', 0) == 0.0)
                )
                if is_empty:
                    offers = ['F3000G100M', 'F1200G50M', 'F3000G200M']
                else:
                    processed_features = preprocess_customer_data(subscriber_data)
                    offers = recommend_nbo(processed_features, som_weights, cluster_offers)
                cursor.execute(
                    "INSERT OR REPLACE INTO customer_recommendations (SubscriberID, recommended_offers) VALUES (?, ?)",
                    (subscriber_id, json.dumps(offers))
                )
                success_count += 1
            except Exception as e:
                print(f"Error processing subscriber {subscriber_id}: {str(e)}")
                error_count += 1
            processed += 1
            if processed % 100 == 0:
                conn.commit()
                print(f"Progress: {processed}/{total_subscribers} subscribers processed")
        conn.commit()
    conn.commit()
    conn.close()
    
    print(f"Processing complete. Successful: {success_count}, Errors: {error_count}")
    return success_count, error_count

# === Export recommendations to CSV (optional) ===
def export_to_csv(db_name="recommendations.db", output_file="customer_recommendations.csv"):
    conn = sqlite3.connect(db_name)
    
    # Read from database into DataFrame
    recommendations_df = pd.read_sql_query("SELECT * FROM customer_recommendations", conn)
    conn.close()
    
    # Parse JSON strings back to lists for better CSV format
    recommendations_df['recommended_offers'] = recommendations_df['recommended_offers'].apply(
        lambda x: ', '.join(json.loads(x))
    )
    
    # Save to CSV
    recommendations_df.to_csv(output_file, index=False)
    print(f"Recommendations exported to {output_file}")

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet('aggregated_subscriber_data2.parquet')

    if 'SubscriberID' not in df.columns:
        raise ValueError("SubscriberID column is missing in the dataset")

    # Process all customers and save to database
    process_all_customers(df)

    # Optionally export to CSV for easier viewing
    export_to_csv()
