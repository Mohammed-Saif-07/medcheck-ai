import joblib
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
 
print("=" * 50)
print("MEDCHECK AI - Testing & Building RAG")
print("=" * 50)
 
print("\n1. Testing ML Model...")
model = joblib.load('models/xgboost_interaction_model.pkl')
print("   ML Model loaded!")
 
print("\n2. Testing Encoder...")
encoder = joblib.load('models/drug_label_encoder.pkl')
print(f"   Encoder: {len(encoder.classes_)} drugs")
 
print("\n3. Testing Metadata...")
with open('models/rag_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f"   Metadata: {len(metadata)} records")
 
print("\n4. Building RAG with NumPy (no FAISS needed)...")
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
 
print("\n5. Creating embeddings (1-2 mins)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
 
np.save('models/rag_embeddings.npy', embeddings)
 
with open('models/rag_documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
 
print(f"   Embeddings saved: {embeddings.shape}")
 
print("\n6. Testing RAG search...")
query = embedder.encode(['warfarin aspirin interaction'])
query = query / np.linalg.norm(query)
scores = np.dot(embeddings, query.T).flatten()
top_indices = np.argsort(scores)[::-1][:3]
 
print("   Search: warfarin + aspirin")
for idx in top_indices:
    m = metadata[idx]
    print(f"   -> {m['drug1']} + {m['drug2']} (score: {scores[idx]:.3f})")
 
print("\n" + "=" * 50)
print("ALL READY!")
print("=" * 50)
 
