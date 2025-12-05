import pandas as pd
import numpy as np
import joblib
import ast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os

# === Create FastAPI app ===
app = FastAPI(
    title="Next Best Offer API",
    description="API for recommending next best offers based on customer features",
    version="1.0.0"
)

# === Define Input Model ===
class CustomerFeatures(BaseModel):
    DpiPolicy: Optional[List[str]] = []
    contentType: Optional[List[str]] = []
    IpProtocol: Optional[List[str]] = []
    appName: Optional[List[str]] = []
    bytesFromClient: float = 0.0
    bytesFromServer: float = 0.0
    sessions_count: int = 0
    transationDuration: float = 0.0

# === Define Output Model ===
class OfferRecommendation(BaseModel):
    recommended_offers: List[str]

# === Load saved models2 ===
@app.on_event("startup")
def load_models2():
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
    
    print("All models2 loaded successfully!")

# === Define the recommendation function ===
def recommend_nbo(customer_features, som_weights, cluster_offers, som_dims=(10, 10)):
    # Ensure correct dimensions
    expected_features = som_weights.shape[2]
    if len(customer_features) != expected_features:
        print(f"WARNING: Feature count mismatch. Got {len(customer_features)}, expected {expected_features}")
        if len(customer_features) > expected_features:
            customer_features = customer_features[:expected_features]
        else:
            temp = np.zeros(expected_features)
            temp[:len(customer_features)] = customer_features
            customer_features = temp
    
    # Reshape to match SOM weights for broadcasting
    customer_features_reshaped = customer_features.reshape(1, 1, -1)
    
    # Calculate distances
    distances = np.linalg.norm(som_weights - customer_features_reshaped, axis=2)
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
    bmu_row, bmu_col = bmu_index
    cluster_id = bmu_row * som_dims[1] + bmu_col
    
    # Get recommended offers for this cluster
    recommended_offers = cluster_offers.get(cluster_id, [])
    
    # Return top 2-3 offers or fallback to default offers if none found
    if not recommended_offers:
        return ['F3000G100M', 'F1200G50M', 'F3000G200M']  # Default offers
    return recommended_offers

# === Preprocess customer data function ===
def preprocess_customer_data(customer: CustomerFeatures) -> np.ndarray:
    # Create a dictionary to convert to DataFrame
    data_dict = {}
    
    # 1. Encode DpiPolicy to bitmap
    if customer.DpiPolicy:
        bitsum = sum(1 << policy_to_bit[p] for p in customer.DpiPolicy if p in policy_to_bit)
        data_dict['DpiPolicy'] = bitsum
    else:
        data_dict['DpiPolicy'] = 0
    
    # 2. Encode contentType to bitmap
    if customer.contentType:
        bitsum = sum(1 << content_to_bit[c] for c in customer.contentType if c in content_to_bit)
        data_dict['contentType'] = bitsum
    else:
        data_dict['contentType'] = 0
    
    # 3. Encode IpProtocol to bitmap
    if customer.IpProtocol:
        bitsum = sum(1 << proto_to_bit[p] for p in customer.IpProtocol if p in proto_to_bit)
        data_dict['IpProtocol'] = bitsum
    else:
        data_dict['IpProtocol'] = 0
    
    # 4. Convert appName to app_count
    data_dict['app_count'] = len(customer.appName) if customer.appName else 0
    
    # 5. Add other numeric features
    data_dict['bytesFromClient'] = max(customer.bytesFromClient, medians.get('bytesFromClient', 1))
    data_dict['bytesFromServer'] = max(customer.bytesFromServer, medians.get('bytesFromServer', 1))
    data_dict['sessions_count'] = customer.sessions_count
    data_dict['transationDuration'] = max(customer.transationDuration, medians.get('transationDuration', 1))
    
    # Create DataFrame with the processed data
    df = pd.DataFrame([data_dict])
    
    # Ensure all necessary columns are present
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only the columns used in training
    df = df[numeric_cols]
    
    # Debug: Print shapes to verify
    print(f"DataFrame shape before scaling: {df.shape}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Apply scaling
    scaled_data = scaler.transform(df)
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"SOM weights shape: {som_weights.shape}")
    
    return scaled_data[0]  # Return first row as numpy array

# === API endpoint for recommendations ===
@app.post("/recommend", response_model=OfferRecommendation)
def get_recommendations(customer: CustomerFeatures):
    try:
        # Check if the customer has no meaningful input
        is_empty = (
            not customer.DpiPolicy and
            not customer.contentType and
            not customer.IpProtocol and
            not customer.appName and
            customer.bytesFromClient == 0.0 and
            customer.bytesFromServer == 0.0 and
            customer.sessions_count == 0 and
            customer.transationDuration == 0.0
        )
        if is_empty:
            # Return default recommendations
            return {"recommended_offers": ['F3000G100M', 'F1200G50M', 'F3000G200M']}
        
        # Otherwise proceed with preprocessing and SOM recommendation
        processed_features = preprocess_customer_data(customer)
        offers = recommend_nbo(processed_features, som_weights, cluster_offers)
        return {"recommended_offers": offers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# === Root endpoint ===
@app.get("/")
def read_root():
    return {"message": "Welcome to the Next Best Offer API", 
            "docs": "/docs",
            "usage": "POST your customer data to /recommend to get offer recommendations"}

# === Run the API server when this script is executed directly ===
if __name__ == "__main__":
    uvicorn.run("Zm:app", host="0.0.0.0", port=8005, reload=True)