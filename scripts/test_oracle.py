import argparse
import redis
import requests
import json
import os
import sys

# Configuration
BODY_HOST = os.getenv("BODY_HOST", "192.168.14.105")
# The deployment output said API is at :8000
ORACLE_API_URL = f"http://{BODY_HOST}:8000/predict" 
REDIS_HOST = os.getenv("REDIS_HOST", BODY_HOST)
REDIS_PORT = 6379

def get_redis_connection():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        return r
    except redis.ConnectionError as e:
        print(f"❌ Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        print(e)
        sys.exit(1)

def fetch_features(r, type_id):
    """
    Fetch raw features from Redis for the given type_id.
    In a real scenario, this might involve aggregating multiple keys.
    For this test, we assume a 'features:<type_id>' key exists or we construct a mock payload.
    """
    key = f"market:data:{type_id}:latest" # Adjust key schema as per project
    data = r.get(key)
    
    if not data:
        print(f"⚠️  No live data found in Redis for TypeID {type_id} (key: {key})")
        print("   Using synthetic features for testing...")
        # Mock features if real data isn't populate yet
        return {
            "type_id": type_id,
            "rsi_14": 42.5,
            "mac_d": 0.05,
            "volatility": 12.3,
            "order_book_imbalance": 0.15,
            "timestamp": "now"
        }
    
    return json.loads(data)

def request_prediction(features):
    """
    Send features to Oracle (Nexus API) for inference.
    """
    try:
        # Adjust payload structure to match what Nexus expects
        payload = {"features": features}
        response = requests.post(ORACLE_API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Oracle Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach Oracle at {ORACLE_API_URL}")
        print(e)
        return None

def main():
    parser = argparse.ArgumentParser(description="Manual Inference Test for EVE Trader Oracle")
    parser.add_argument("type_id", type=int, help="EVE Online Type ID (e.g., 34 for Tritanium)")
    args = parser.parse_args()

    print(f"🔍 Testing Oracle for Item: {args.type_id}")
    print(f"   Redis Bridge: {REDIS_HOST}")
    print(f"   Oracle API:   {ORACLE_API_URL}")

    # 1. Connect to Redis
    r = get_redis_connection()
    print("✅ Redis Connected")

    # 2. Fetch Features
    features = fetch_features(r, args.type_id)
    print(f"✅ Features Acquired: {list(features.keys())}")

    # 3. Request Prediction
    print("⏳ Requesting Prediction...")
    result = request_prediction(features)

    # 4. Display Results
    if result:
        # Adapt keys based on actual API response
        target_price = result.get('predicted_price', result.get('target', 0.0))
        confidence = result.get('confidence', 0.0)
        
        print("\n" + "="*40)
        print(f"🔮 ORACLE PREDICTION (T+60m)")
        print(f"   Target Price:     {target_price:,.2f} ISK")
        print(f"   Confidence Score: {confidence:.2%}")
        print("="*40 + "\n")
    else:
        print("\n❌ Inference Failed.")

if __name__ == "__main__":
    main()
