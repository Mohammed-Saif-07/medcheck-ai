"""
MedCheck RAG - Vector Database Builder

Converts FDA drug labels into embeddings
Builds FAISS vector database for fast retrieval

Uses sentence-transformers (FREE and lightweight!)
"""

import json
import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict
import re

class VectorDBBuilder:
    """
    Build vector database from FDA drug labels
    Uses simple TF-IDF for lightweight implementation
    (Can upgrade to sentence-transformers later!)
    """
    
    def __init__(self, labels_file="data/raw/fda_labels_combined.json"):
        self.labels_file = Path(labels_file)
        self.output_dir = Path("data/processed/vector_db")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents = []
        self.metadata = []
        
        print("🔧 Vector Database Builder Initialized")
        print(f"📁 Labels file: {labels_file}")
        print(f"📁 Output: {self.output_dir}")
        print()
    
    def load_labels(self):
        """Load FDA drug labels"""
        
        print("📂 Loading FDA labels...")
        
        if not self.labels_file.exists():
            print(f"❌ Labels file not found: {self.labels_file}")
            print("   Run fda_downloader.py first!")
            return False
        
        with open(self.labels_file, 'r') as f:
            labels_data = json.load(f)
        
        print(f"   ✅ Loaded {len(labels_data)} drug labels")
        
        # Extract text chunks from each label
        for label in labels_data:
            drug_name = label['drug_name']
            interactions_text = label.get('interactions_text', '')
            
            if interactions_text:
                # Split into chunks
                chunks = self.create_chunks(interactions_text, drug_name)
                
                for chunk in chunks:
                    self.documents.append(chunk['text'])
                    self.metadata.append({
                        'drug_name': drug_name,
                        'generic_name': label.get('generic_name', drug_name),
                        'section': chunk['section'],
                        'source': 'FDA OpenFDA'
                    })
        
        print(f"   ✅ Created {len(self.documents)} text chunks")
        print()
        
        return True
    
    def create_chunks(self, text: str, drug_name: str, chunk_size=500) -> List[Dict]:
        """Split text into chunks with section headers"""
        
        chunks = []
        
        # Split by sections
        sections = text.split('\n\n')
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip very short sections
                continue
            
            # Identify section type
            section_type = "General"
            if "DRUG INTERACTIONS" in section:
                section_type = "Drug Interactions"
            elif "WARNINGS" in section:
                section_type = "Warnings"
            elif "CONTRAINDICATIONS" in section:
                section_type = "Contraindications"
            elif "ADVERSE REACTIONS" in section:
                section_type = "Adverse Reactions"
            
            # Add drug name context
            chunk_text = f"Drug: {drug_name}\nSection: {section_type}\n\n{section}"
            
            chunks.append({
                'text': chunk_text,
                'section': section_type
            })
        
        return chunks
    
    def build_simple_index(self):
        """Build simple keyword-based index (lightweight alternative to embeddings)"""
        
        print("🏗️  Building keyword index...")
        
        # Create keyword index
        keyword_index = {}
        
        for i, doc in enumerate(self.documents):
            # Extract keywords
            words = re.findall(r'\b[a-z]{4,}\b', doc.lower())
            
            for word in set(words):
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(i)
        
        # Save index
        index_file = self.output_dir / "keyword_index.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump(keyword_index, f)
        
        print(f"   ✅ Keyword index built: {len(keyword_index)} unique keywords")
        print()
    
    def build(self):
        """Main build pipeline"""
        
        print("="*70)
        print("🏗️  BUILDING VECTOR DATABASE")
        print("="*70)
        print()
        
        # Load labels
        if not self.load_labels():
            return False
        
        # Build index
        self.build_simple_index()
        
        # Save documents and metadata
        docs_file = self.output_dir / "documents.json"
        with open(docs_file, 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f, indent=2)
        
        print(f"💾 Documents saved to: {docs_file}")
        print()
        
        print("="*70)
        print("✅ VECTOR DATABASE BUILD COMPLETE!")
        print("="*70)
        print()
        print("📊 Database Stats:")
        print(f"   Total chunks: {len(self.documents)}")
        print(f"   Unique drugs: {len(set(m['drug_name'] for m in self.metadata))}")
        print()
        print("🎯 Next step: Test RAG retrieval!")
        print("   Run: python src/rag/rag_retriever.py")
        print()
        
        return True


def main():
    """Main build workflow"""
    
    print("\n" + "="*70)
    print("💊 MedCheck RAG - Vector Database Builder")
    print("="*70)
    print()
    
    builder = VectorDBBuilder()
    builder.build()


if __name__ == "__main__":
    main()
