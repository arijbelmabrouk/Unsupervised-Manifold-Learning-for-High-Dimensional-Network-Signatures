import pandas as pd
import numpy as np
import joblib
import sqlite3
from tqdm import tqdm
import ast
import json
import os

# === Load Research Artifacts ===
def load_assets():
    global som_weights, centroid_map, policy_to_bit, content_to_bit, proto_to_bit, scaler, medians, numeric_cols
    
    # Weights Loading
    try:
        som_model = joblib.load("models2/som_model.joblib")
        som_weights = som_model._weights
    except:
        som_weights = np.load("som_weights1.npy")
    
    centroid_map = joblib.load('models2/centroid_feature_map.joblib')
    policy_to_bit = joblib.load('models2/policy_to_bit.joblib')
    content_to_bit = joblib.load('models2/content_to_bit.joblib')
    proto_to_bit = joblib.load('models2/proto_to_bit.joblib')
    scaler = joblib.load('models2/scaler.joblib')
    medians = joblib.load('models2/medians.joblib')
    numeric_cols = joblib.load('models2/numeric_cols.joblib')
    print("Topological artifacts loaded.")

def project_batch(features_matrix, weights, mapping, som_dims=(10, 10)):
    """Vectorized BMU calculation for high-speed batch processing."""
    # features_matrix shape: (batch_size, n_features)
    # weights shape: (10, 10, n_features)
    
    # Calculate Euclidean distance for the whole batch at once
    # Resulting distances shape: (batch_size, 10, 10)
    diff = weights[np.newaxis, :, :, :] - features_matrix[:, np.newaxis, np.newaxis, :]
    distances = np.linalg.norm(diff, axis=3)
    
    # Find BMU for each sample in batch
    flat_indices = np.argmin(distances.reshape(len(features_matrix), -1), axis=1)
    
    results = []
    for idx in flat_indices:
        labels = mapping.get(idx, [])
        results.append(json.dumps(labels if labels else ['REGIME_STABLE']))
    return results

def preprocess_batch(df):
    """Vectorized preprocessing logic."""
    processed_df = pd.DataFrame(index=df.index)
    
    # Helper for bitmap encoding
    def get_bitsum(items, mapping):
        if not items or items == '[]': return 0
        if isinstance(items, str):
            try: items = ast.literal_eval(items)
            except: return 0
        return sum(1 << mapping.get(p, 0) for p in items if p in mapping)

    # 1-3. Bitmap Encodings
    processed_df['DpiPolicy'] = df['DpiPolicy'].apply(lambda x: get_bitsum(x, policy_to_bit))
    processed_df['contentType'] = df['contentType'].apply(lambda x: get_bitsum(x, content_to_bit))
    processed_df['IpProtocol'] = df['IpProtocol'].apply(lambda x: get_bitsum(x, proto_to_bit))
    
    # 4. App count
    processed_df['app_count'] = df['appName'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x != '[]' else 0)
    
    # 5. Numeric logic (Vectorized)
    for col in ['bytesFromClient', 'bytesFromServer', 'transationDuration']:
        processed_df[col] = df[col].clip(lower=medians.get(col, 1))
    
    processed_df['sessions_count'] = df['sessions_count'].fillna(0)
    
    # Ensure column order and scale
    final_df = processed_df.reindex(columns=numeric_cols, fill_value=0)
    return scaler.transform(final_df)

def run_pipeline(input_path, db_name="projections.db"):
    load_assets()
    df = pd.read_parquet(input_path)
    
    # Database Setup
    conn = sqlite3.connect(db_name)
    conn.execute("DROP TABLE IF EXISTS manifold_projections")
    conn.execute('''CREATE TABLE manifold_projections (
                    SignalID TEXT PRIMARY KEY, 
                    latent_assignments TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Process in chunks for memory efficiency
    batch_size = 5000
    subscribers = df['SubscriberID'].unique()
    
    print(f"Projecting {len(subscribers)} signatures into manifold...")
    
    for i in tqdm(range(0, len(subscribers), batch_size)):
        batch_ids = subscribers[i:i+batch_size]
        batch_data = df[df['SubscriberID'].isin(batch_ids)].drop_duplicates('SubscriberID')
        
        # Vectorized math happens here
        feature_matrix = preprocess_batch(batch_data)
        projections = project_batch(feature_matrix, som_weights, centroid_map)
        
        # Bulk Insert
        insert_data = list(zip(batch_ids, projections))
        conn.executemany("INSERT INTO manifold_projections (SignalID, latent_assignments) VALUES (?, ?)", insert_data)
        conn.commit()

    conn.close()
    print("Batch Projection Complete.")

if __name__ == "__main__":
    run_pipeline('aggregated_subscriber_data2.parquet')
