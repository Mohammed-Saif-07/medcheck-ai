import pickle
import numpy as np
import os

print("=" * 50)
print("STEP 1: Testing Metadata Only (No XGBoost)")
print("=" * 50)

with open('models/rag_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f"Metadata: {len(metadata)} records")

print("\nSample data:")
for m in metadata[:5]:
    print(f"  {m['drug1']} + {m['drug2']} = {m['severity']} ({m['total_reports']} reports)")

print("\nSTEP 1 PASSED!")
