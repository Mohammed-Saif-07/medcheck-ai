import pickle
import numpy as np
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import os

print("=" * 50)
print("MEDCHECK AI - Testing Models")
print("=" * 50)

print("\n1. Testing ML Model (JSON format)...")
model = XGBClassifier()
if os.path.exists('models/xgboost_model.json'):
    model.load_model('models/xgboost_model.json')
    print("   ML Model loaded from JSON!")
else:
    print("   xgboost_model.json not found yet!")
    print("   Run the Colab cell first to export it.")

print("\n2. Testing Encoder...")
try:
    import joblib
    encoder = joblib.load('models/drug_label_encoder.pkl')
    print(f"   Encoder: {len(encoder.classes_)} drugs")
except Exception as e:
    print(f"   Encoder failed: {e}")
    print("   Will rebuild encoder locally.")

print("\n3. Testing Metadata...")
with open('models/rag_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f"   Metadata: {len(metadata)} records")

print("\n4. Building RAG with NumPy...")
documents = []
for m in metadata:
    doc = (
        f"{m['drug1'].title()} and {m['drug2'].title()} interaction. "
        f"Severity: {m['severity']}. "
        f"FDA Reports: {m['total_reports']}. "
        f"Serious: {m['serious_reports']}. "
        f"Reactions: {m.get('top_reactions', 'N/A')}."
    )
    documents.append(doc)

print(f"   Documents: {len(documents)}")

print("\n5. Creating embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

np.save('models/rag_embeddings.npy', embeddings)

with open('models/rag_documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print(f"   Saved: {embeddings.shape}")

print("\n6. Testing RAG search...")
query = embedder.encode(['warfarin aspirin interaction'])
query = query / np.linalg.norm(query)
scores = np.dot(embeddings, query.T).flatten()
top_idx = np.argsort(scores)[::-1][:3]

for idx in top_idx:
    m = metadata[idx]
    print(f"   {m['drug1']} + {m['drug2']} (score: {scores[idx]:.3f})")

print("\n" + "=" * 50)
print("DONE! All files ready!")
print("=" * 50)
