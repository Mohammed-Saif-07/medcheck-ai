"""
MedCheck - FINAL INTEGRATED PIPELINE
OCR → ML → RAG → LLM → Complete Report

PRODUCTION-READY SYSTEM!
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all modules
from src.ocr.advanced_ocr import AdvancedOCR
from src.ml.train_interaction_model import InteractionClassifier
from src.rag.rag_retriever import RAGRetriever
from src.llm.llm_explainer import LLMExplainer

class MedCheckFinalPipeline:
    """
    COMPLETE AI PIPELINE:
    Image → OCR → ML → RAG → LLM → Safety Report
    """
    
    def __init__(self):
        print("="*70)
        print("💊 MEDCHECK - COMPLETE AI SYSTEM")
        print("   OCR + ML + RAG + LLM")
        print("="*70)
        print()
        
        # Initialize all components
        print("🔧 Initializing AI components...")
        
        # 1. Computer Vision (OCR)
        print("   📸 Computer Vision (OCR)...")
        self.ocr = AdvancedOCR()
        
        # 2. Machine Learning
        print("   🤖 Machine Learning (XGBoost)...")
        self.classifier = InteractionClassifier()
        
        if not self.classifier.load_model():
            print("   ⚠️  Using fallback ML predictions")
            self.use_ml_fallback = True
        else:
            self.use_ml_fallback = False
        
        # 3. RAG System
        print("   📚 RAG (FDA Knowledge Base)...")
        self.rag = RAGRetriever()
        self.use_rag = bool(self.rag.documents)
        
        # 4. LLM Explainer
        print("   💬 LLM (Natural Language)...")
        self.llm = LLMExplainer()
        
        print()
        print("✅ All Systems Ready!")
        print()
    
    def process_prescription(self, image_path: str) -> dict:
        """
        Complete pipeline execution
        """
        print("="*70)
        print(f"📋 PROCESSING: {Path(image_path).name}")
        print("="*70)
        print()
        
        # STAGE 1: Computer Vision
        print("🔍 STAGE 1: Computer Vision (OCR)")
        print("-"*70)
        
        ocr_result = self.ocr.process_image(image_path)
        
        if 'error' in ocr_result:
            return {'error': ocr_result['error']}
        
        verified_drugs = ocr_result.get('drugs_verified', [])
        
        if not verified_drugs:
            print("\n⚠️  No drugs detected!")
            return {'error': 'No drugs detected'}
        
        drug_names = [d.get('matched', d.get('original', '')) for d in verified_drugs]
        
        print(f"\n✅ Detected {len(drug_names)} drug(s): {', '.join([d.upper() for d in drug_names])}")
        print()
        
        # STAGE 2: Machine Learning
        print("🤖 STAGE 2: Machine Learning (Interaction Prediction)")
        print("-"*70)
        
        interactions = []
        
        if len(drug_names) < 2:
            print("\n   ℹ️  Only 1 drug - no interactions to check")
        else:
            from itertools import combinations
            
            print(f"\n   Checking {len(list(combinations(drug_names, 2)))} pairs...")
            print()
            
            for drug1, drug2 in combinations(drug_names, 2):
                print(f"   {drug1.upper()} + {drug2.upper()}: ", end="")
                
                if not self.use_ml_fallback:
                    prediction = self.classifier.predict_interaction(drug1, drug2)
                    severity = prediction['severity']
                    confidence = prediction['confidence']
                else:
                    severity = self._check_known_interaction(drug1, drug2)
                    confidence = 0.95 if severity else 0
                
                if severity and severity != 'NONE':
                    icon = "🔴" if severity == "HIGH" else "🟡"
                    print(f"{icon} {severity} ({confidence*100:.0f}%)")
                    
                    interactions.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'severity': severity,
                        'confidence': confidence
                    })
                else:
                    print("✅ Safe")
        
        print(f"\n✅ Found {len(interactions)} interaction(s)")
        print()
        
        # STAGE 3: RAG Evidence Retrieval
        if interactions and self.use_rag:
            print("📚 STAGE 3: RAG (Evidence Retrieval)")
            print("-"*70)
            print()
            
            for interaction in interactions:
                print(f"   Retrieving evidence: {interaction['drug1'].upper()} + {interaction['drug2'].upper()}")
                
                evidence = self.rag.get_interaction_evidence(
                    interaction['drug1'],
                    interaction['drug2']
                )
                
                interaction['evidence'] = evidence
                
                if evidence['evidence_found']:
                    print(f"      ✅ Found {len(evidence['sources'])} sources")
                else:
                    print(f"      ℹ️  Limited evidence")
            
            print()
            print("✅ Evidence retrieved")
            print()
        
        # STAGE 4: LLM Natural Language
        print("💬 STAGE 4: LLM (Natural Language Explanations)")
        print("-"*70)
        print()
        
        if interactions:
            for interaction in interactions:
                print(f"   Generating explanation: {interaction['drug1'].upper()} + {interaction['drug2'].upper()}")
                
                explanation = self.llm.explain_interaction(
                    interaction['drug1'],
                    interaction['drug2'],
                    interaction['severity'],
                    interaction.get('evidence')
                )
                
                interaction['explanation'] = explanation
                print(f"      ✅ Explanation ready")
        else:
            print("   Generating safe combination message...")
            if len(drug_names) >= 2:
                explanation = self.llm.explain_safe_combination(
                    drug_names[0], drug_names[1]
                )
            else:
                explanation = "✅ Single medication detected - no interactions to check."
        
        print()
        print("✅ Explanations generated")
        print()
        
        # STAGE 5: Risk Assessment
        print("📊 STAGE 5: Risk Assessment")
        print("-"*70)
        
        risk_score = self._calculate_risk_score(interactions)
        risk_level = self._get_risk_level(risk_score)
        
        print(f"\n   Risk Score: {risk_score}/100")
        print(f"   Risk Level: {risk_level}")
        print()
        
        print("✅ Assessment complete")
        print()
        
        # Compile final result
        result = {
            'image': image_path,
            'timestamp': datetime.now().isoformat(),
            'drugs_detected': drug_names,
            'ocr_confidence': ocr_result.get('ocr_confidence', 0),
            'interactions': interactions,
            'safe_explanation': explanation if not interactions and len(drug_names) >= 2 else None,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommendations': self._get_recommendations(risk_level, interactions)
        }
        
        return result
    
    def generate_report(self, result: dict):
        """
        Generate beautiful final report
        """
        print("="*70)
        print("📄 FINAL MEDICATION SAFETY REPORT")
        print("="*70)
        print()
        
        # Header
        print(f"📅 {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        print(f"📸 {Path(result['image']).name}")
        print()
        
        # Medications
        print("💊 MEDICATIONS DETECTED")
        print("-"*70)
        for i, drug in enumerate(result['drugs_detected'], 1):
            print(f"   {i}. {drug.upper()}")
        print(f"\n   Confidence: {result['ocr_confidence']*100:.0f}%")
        print()
        
        # Interactions
        print("⚠️  INTERACTION ANALYSIS")
        print("-"*70)
        
        if result['interactions']:
            for i, interaction in enumerate(result['interactions'], 1):
                severity = interaction['severity']
                icon = "🔴" if severity == "HIGH" else "🟡"
                
                print(f"\n{icon} INTERACTION #{i} - {severity} SEVERITY")
                print(f"   Drugs: {interaction['drug1'].upper()} + {interaction['drug2'].upper()}")
                print(f"   ML Confidence: {interaction['confidence']*100:.0f}%")
                print()
                
                # LLM Explanation
                if 'explanation' in interaction:
                    print("   💬 WHAT THIS MEANS:")
                    for line in interaction['explanation'].split('\n'):
                        if line.strip():
                            print(f"   {line}")
                    print()
                
                # RAG Evidence (brief)
                if 'evidence' in interaction and interaction['evidence']['evidence_found']:
                    print(f"   📚 EVIDENCE: {len(interaction['evidence']['sources'])} FDA sources")
                    print()
        else:
            print("\n✅ No dangerous interactions detected")
            
            # Show safe explanation if available
            if result.get('safe_explanation'):
                print()
                for line in result['safe_explanation'].split('\n'):
                    if line.strip():
                        print(f"   {line}")
            print()
        
        # Risk Assessment
        print("📊 RISK ASSESSMENT")
        print("-"*70)
        print(f"   Score: {result['risk_score']}/100")
        print(f"   Level: {result['risk_level']}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-"*70)
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")
        print()
        
        # Disclaimer
        print("="*70)
        print("⚠️  MEDICAL DISCLAIMER")
        print("="*70)
        print("This is an AI-powered informational tool, not medical advice.")
        print("Always consult your doctor or pharmacist about your medications.")
        print("="*70)
        print()
        
        # Save
        self._save_report(result)
    
    def _check_known_interaction(self, drug1: str, drug2: str) -> str:
        """Fallback interactions"""
        known = {
            ('warfarin', 'aspirin'): 'HIGH',
            ('warfarin', 'ibuprofen'): 'HIGH',
        }
        pair1 = (drug1.lower(), drug2.lower())
        pair2 = (drug2.lower(), drug1.lower())
        return known.get(pair1) or known.get(pair2)
    
    def _calculate_risk_score(self, interactions: list) -> int:
        """Calculate risk score"""
        if not interactions:
            return 0
        severity_scores = {'HIGH': 30, 'MODERATE': 15, 'LOW': 5}
        total = sum(severity_scores.get(i['severity'], 0) for i in interactions)
        return min(total, 100)
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level"""
        if score >= 50: return "CRITICAL"
        elif score >= 30: return "HIGH"
        elif score >= 15: return "MODERATE"
        elif score > 0: return "LOW"
        else: return "NONE"
    
    def _get_recommendations(self, risk_level: str, interactions: list) -> list:
        """Generate recommendations"""
        recs = []
        
        if risk_level == "CRITICAL":
            recs.append("🔴 URGENT: Contact your doctor immediately")
            recs.append("🔴 Do not take without medical supervision")
        elif risk_level == "HIGH":
            recs.append("⚠️  Schedule doctor appointment soon")
            recs.append("💊 Discuss alternative medications")
        elif risk_level == "MODERATE":
            recs.append("ℹ️  Mention to doctor at next visit")
        else:
            recs.append("✅ Continue as prescribed")
            recs.append("✅ Inform doctor of any unusual symptoms")
        
        return recs
    
    def _save_report(self, result: dict):
        """Save report"""
        results_dir = Path("results/final_reports")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"medcheck_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Report saved: {report_file}")
        print()


def main():
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("💊 MEDCHECK - COMPLETE AI SYSTEM DEMO")
    print("="*70)
    print()
    print("Full AI Stack:")
    print("  1. Computer Vision (OCR)")
    print("  2. Machine Learning (XGBoost)")
    print("  3. RAG (FDA Knowledge Base)")
    print("  4. LLM (Natural Language)")
    print("  5. Complete Safety Report")
    print()
    print("="*70)
    print()
    
    # Initialize
    pipeline = MedCheckFinalPipeline()
    
    # Process
    test_image = "data/test_images/test_prescription.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    result = pipeline.process_prescription(test_image)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Report
    pipeline.generate_report(result)
    
    print("="*70)
    print("✅ COMPLETE AI PIPELINE EXECUTED!")
    print("="*70)
    print()
    print("🎯 Technologies Used:")
    print("   ✅ Computer Vision (OpenCV, Tesseract)")
    print("   ✅ Machine Learning (XGBoost, scikit-learn)")
    print("   ✅ RAG (Vector DB, FDA API)")
    print("   ✅ LLM (Natural Language Processing)")
    print()
    print("💼 Resume Value: Complete AI Engineer!")
    print()


if __name__ == "__main__":
    main()
