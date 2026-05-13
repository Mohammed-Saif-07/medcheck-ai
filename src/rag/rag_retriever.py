"""
MedCheck RAG - Retrieval System

Retrieves relevant information from vector database
Provides evidence with citations for drug interactions
"""

import json
import pickle
from pathlib import Path
import re
from typing import List, Dict
from collections import Counter

class RAGRetriever:
    """
    RAG Retrieval System
    Searches vector database for relevant drug interaction evidence
    """
    
    def __init__(self):
        self.db_dir = Path("data/processed/vector_db")
        
        print("🔍 RAG Retriever Initialized")
        
        # Load database
        self.load_database()
    
    def load_database(self):
        """Load vector database and documents"""
        
        # Load documents
        docs_file = self.db_dir / "documents.json"
        
        if not docs_file.exists():
            print(f"❌ Database not found: {docs_file}")
            print("   Run build_vector_db.py first!")
            self.documents = []
            self.metadata = []
            self.keyword_index = {}
            return False
        
        with open(docs_file, 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
        
        # Load keyword index
        index_file = self.db_dir / "keyword_index.pkl"
        with open(index_file, 'rb') as f:
            self.keyword_index = pickle.load(f)
        
        print(f"   ✅ Loaded {len(self.documents)} documents")
        print(f"   ✅ Loaded keyword index")
        print()
        
        return True
    
    def search(self, query: str, drug1: str = None, drug2: str = None, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents
        """
        
        if not self.documents:
            return []
        
        # Build search query
        search_terms = []
        
        if drug1:
            search_terms.append(drug1.lower())
        if drug2:
            search_terms.append(drug2.lower())
        
        # Add query terms
        query_words = re.findall(r'\b[a-z]{4,}\b', query.lower())
        search_terms.extend(query_words)
        
        # Find matching documents
        doc_scores = Counter()
        
        for term in search_terms:
            if term in self.keyword_index:
                for doc_id in self.keyword_index[term]:
                    doc_scores[doc_id] += 1
        
        # Get top-k documents
        top_docs = []
        
        for doc_id, score in doc_scores.most_common(top_k):
            top_docs.append({
                'text': self.documents[doc_id],
                'metadata': self.metadata[doc_id],
                'relevance_score': score
            })
        
        return top_docs
    
    def get_interaction_evidence(self, drug1: str, drug2: str) -> Dict:
        """
        Get evidence for drug interaction
        """
        
        print(f"\n🔍 Searching for: {drug1.upper()} + {drug2.upper()}")
        print("-"*70)
        
        # Search for evidence
        results = self.search(
            query=f"{drug1} {drug2} interaction",
            drug1=drug1,
            drug2=drug2,
            top_k=5
        )
        
        if not results:
            print("   ℹ️  No specific evidence found in database")
            return {
                'drug1': drug1,
                'drug2': drug2,
                'evidence_found': False,
                'sources': []
            }
        
        print(f"   ✅ Found {len(results)} relevant sources")
        
        # Format evidence
        evidence = {
            'drug1': drug1,
            'drug2': drug2,
            'evidence_found': True,
            'sources': []
        }
        
        for i, result in enumerate(results, 1):
            source = {
                'citation_number': i,
                'drug_name': result['metadata']['drug_name'],
                'section': result['metadata']['section'],
                'text': result['text'],
                'source': result['metadata']['source'],
                'relevance': result['relevance_score']
            }
            evidence['sources'].append(source)
            
            print(f"\n   [{i}] {result['metadata']['drug_name']} - {result['metadata']['section']}")
            print(f"       Relevance: {result['relevance_score']}")
        
        print()
        
        return evidence
    
    def format_evidence_report(self, evidence: Dict) -> str:
        """
        Format evidence into readable report
        """
        
        if not evidence['evidence_found']:
            return f"""
⚠️  Limited Evidence Available

No specific interaction data found in FDA database for 
{evidence['drug1'].upper()} + {evidence['drug2'].upper()}.

This does not mean the combination is safe. Always consult 
your doctor or pharmacist about potential interactions.
"""
        
        report = f"""
📚 EVIDENCE FROM FDA DATABASE

Drug Interaction: {evidence['drug1'].upper()} + {evidence['drug2'].upper()}

"""
        
        for source in evidence['sources']:
            report += f"""
[{source['citation_number']}] {source['drug_name'].upper()} - {source['section']}

{self._extract_key_sentences(source['text'], evidence['drug1'], evidence['drug2'])}

Source: {source['source']}

"""
        
        report += """
---
📌 Note: This information is extracted from FDA drug labels and 
should not replace professional medical advice.
"""
        
        return report
    
    def _extract_key_sentences(self, text: str, drug1: str, drug2: str, max_sentences: int = 3) -> str:
        """Extract most relevant sentences from text"""
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        # Score sentences by relevance
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            score = 0
            sentence_lower = sentence.lower()
            
            # Check for drug mentions
            if drug1.lower() in sentence_lower:
                score += 2
            if drug2.lower() in sentence_lower:
                score += 2
            
            # Check for interaction keywords
            keywords = ['interaction', 'concurrent', 'combination', 
                       'together', 'risk', 'warning', 'caution']
            
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += 1
            
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Get top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences) + '.' if top_sentences else text[:500]


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo the RAG retrieval system"""
    
    print("\n" + "="*70)
    print("💊 MedCheck RAG - Retrieval Demo")
    print("="*70)
    print()
    
    retriever = RAGRetriever()
    
    if not retriever.documents:
        print("❌ No database loaded. Run build_vector_db.py first!")
        return
    
    # Test queries
    test_pairs = [
        ('warfarin', 'aspirin'),
        ('lisinopril', 'metformin'),
        ('ibuprofen', 'aspirin'),
    ]
    
    for drug1, drug2 in test_pairs:
        evidence = retriever.get_interaction_evidence(drug1, drug2)
        
        print("="*70)
        print(retriever.format_evidence_report(evidence))
        print("="*70)
        print()
        
        input("Press Enter for next example...")
        print()


if __name__ == "__main__":
    demo()
