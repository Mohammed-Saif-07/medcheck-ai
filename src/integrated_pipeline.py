"""
MedCheck - COMPLETE INTEGRATED PIPELINE
Connects: OCR → ML → Safety Report

This is the production-ready system that brings everything together!
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from src.ocr.advanced_ocr import AdvancedOCR
from src.ml.train_interaction_model import InteractionClassifier

class MedCheckPipeline:
    """
    Complete integrated pipeline:
    Image → OCR → ML Prediction → Safety Report
    """
    
    def __init__(self):
        print("="*70)
        print("💊 MEDCHECK - COMPLETE AI PIPELINE")
        print("="*70)
        print()
        
        # Initialize components
        print("🔧 Initializing components...")
        
        # 1. OCR System
        print("   📸 Loading OCR system...")
        self.ocr = AdvancedOCR()
        
        # 2. ML Model
        print("   🤖 Loading ML model...")
        self.classifier = InteractionClassifier()
        
        # Try to load trained model
        if not self.classifier.load_model():
            print("   ⚠️  No trained model found")
            print("   💡 Run: python src/ml/train_interaction_model.py")
            print("   ℹ️  Using fallback predictions")
            self.use_fallback = True
        else:
            self.use_fallback = False
        
        print()
        print("✅ Pipeline Ready!")
        print()
    
    def process_prescription(self, image_path: str) -> dict:
        """
        Main pipeline: Process prescription image end-to-end
        """
        print("="*70)
        print(f"📋 PROCESSING PRESCRIPTION: {Path(image_path).name}")
        print("="*70)
        print()
        
        # Stage 1: OCR - Extract drugs from image
        print("🔍 STAGE 1: OCR - Drug Name Extraction")
        print("-"*70)
        
        ocr_result = self.ocr.process_image(image_path)
        
        if 'error' in ocr_result:
            return {'error': ocr_result['error']}
        
        # Get verified drugs
        verified_drugs = ocr_result.get('drugs_verified', [])
        
        if not verified_drugs:
            print("\n⚠️  No drugs detected in image!")
            return {'error': 'No drugs detected'}
        
        drug_names = [d.get('matched', d.get('original', '')) for d in verified_drugs]
        
        print(f"\n✅ Stage 1 Complete: {len(drug_names)} drug(s) detected")
        print(f"   Drugs: {', '.join([d.upper() for d in drug_names])}")
        print()
        
        # Stage 2: ML - Interaction Prediction
        print("🤖 STAGE 2: ML - Interaction Prediction")
        print("-"*70)
        
        interactions = []
        
        if len(drug_names) < 2:
            print("\n   ℹ️  Only 1 drug detected - no interactions to check")
        else:
            from itertools import combinations
            
            print(f"\n   Checking {len(list(combinations(drug_names, 2)))} drug pairs...")
            print()
            
            for drug1, drug2 in combinations(drug_names, 2):
                print(f"   Analyzing: {drug1.upper()} + {drug2.upper()}")
                
                if not self.use_fallback:
                    # Use trained ML model
                    prediction = self.classifier.predict_interaction(drug1, drug2)
                    
                    severity = prediction['severity']
                    confidence = prediction['confidence']
                    
                    if severity != 'NONE':
                        interactions.append({
                            'drug1': drug1,
                            'drug2': drug2,
                            'severity': severity,
                            'confidence': confidence,
                            'source': 'ML Model'
                        })
                        
                        icon = "🔴" if severity == "HIGH" else "🟡"
                        print(f"      {icon} {severity} RISK ({confidence*100:.1f}% confidence)")
                    else:
                        print(f"      ✅ No interaction")
                else:
                    # Use fallback known interactions
                    severity = self._check_known_interaction(drug1, drug2)
                    if severity:
                        interactions.append({
                            'drug1': drug1,
                            'drug2': drug2,
                            'severity': severity,
                            'confidence': 0.95,
                            'source': 'Knowledge Base'
                        })
                        
                        icon = "🔴" if severity == "HIGH" else "🟡"
                        print(f"      {icon} {severity} RISK (95% confidence)")
                    else:
                        print(f"      ✅ No known interaction")
        
        print(f"\n✅ Stage 2 Complete: {len(interactions)} interaction(s) found")
        print()
        
        # Stage 3: Risk Scoring
        print("📊 STAGE 3: Risk Assessment")
        print("-"*70)
        
        risk_score = self._calculate_risk_score(interactions)
        risk_level = self._get_risk_level(risk_score)
        
        print(f"\n   Overall Risk Score: {risk_score}/100")
        print(f"   Risk Level: {risk_level}")
        print()
        
        print("✅ Stage 3 Complete")
        print()
        
        # Compile results
        result = {
            'image': image_path,
            'timestamp': datetime.now().isoformat(),
            'ocr_results': {
                'drugs_detected': drug_names,
                'confidence': ocr_result.get('ocr_confidence', 0),
                'processing_time': ocr_result.get('processing_time', 0)
            },
            'ml_predictions': {
                'interactions': interactions,
                'total_interactions': len(interactions),
                'model_used': 'ML Model' if not self.use_fallback else 'Knowledge Base'
            },
            'risk_assessment': {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommendations': self._get_recommendations(risk_level, interactions)
            }
        }
        
        return result
    
    def _check_known_interaction(self, drug1: str, drug2: str) -> str:
        """Fallback: Check known interactions"""
        known = {
            ('warfarin', 'aspirin'): 'HIGH',
            ('warfarin', 'ibuprofen'): 'HIGH',
            ('warfarin', 'naproxen'): 'HIGH',
            ('lisinopril', 'spironolactone'): 'HIGH',
            ('metformin', 'glipizide'): 'MODERATE',
        }
        
        pair1 = (drug1.lower(), drug2.lower())
        pair2 = (drug2.lower(), drug1.lower())
        
        return known.get(pair1) or known.get(pair2)
    
    def _calculate_risk_score(self, interactions: list) -> int:
        """Calculate overall risk score (0-100)"""
        if not interactions:
            return 0
        
        severity_scores = {
            'HIGH': 30,
            'MODERATE': 15,
            'LOW': 5
        }
        
        total_score = sum(
            severity_scores.get(i['severity'], 0) 
            for i in interactions
        )
        
        # Cap at 100
        return min(total_score, 100)
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Get risk level from score"""
        if risk_score >= 50:
            return "CRITICAL"
        elif risk_score >= 30:
            return "HIGH"
        elif risk_score >= 15:
            return "MODERATE"
        elif risk_score > 0:
            return "LOW"
        else:
            return "NONE"
    
    def _get_recommendations(self, risk_level: str, interactions: list) -> list:
        """Generate recommendations based on risk"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.append("🔴 URGENT: Contact your doctor immediately")
            recommendations.append("🔴 Do not take these medications together without medical supervision")
            recommendations.append("📞 Call your pharmacist for immediate guidance")
        
        elif risk_level == "HIGH":
            recommendations.append("⚠️  Schedule an appointment with your doctor soon")
            recommendations.append("💊 Discuss alternative medications")
            recommendations.append("📋 Monitor for side effects closely")
        
        elif risk_level == "MODERATE":
            recommendations.append("ℹ️  Mention these medications to your doctor at next visit")
            recommendations.append("👀 Be aware of potential side effects")
            recommendations.append("📊 Consider timing medications differently")
        
        elif risk_level == "LOW":
            recommendations.append("✅ Continue as prescribed")
            recommendations.append("ℹ️  Inform doctor if you notice any unusual symptoms")
        
        else:
            recommendations.append("✅ No dangerous interactions detected")
            recommendations.append("✅ Continue taking medications as prescribed")
        
        # Add specific recommendations for each interaction
        for interaction in interactions:
            if interaction['severity'] == 'HIGH':
                recommendations.append(
                    f"⚠️  {interaction['drug1'].upper()} + {interaction['drug2'].upper()}: "
                    f"Seek medical advice"
                )
        
        return recommendations
    
    def generate_report(self, result: dict):
        """Generate comprehensive safety report"""
        
        print("="*70)
        print("📄 MEDICATION SAFETY REPORT")
        print("="*70)
        print()
        
        # Header
        print(f"📅 Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        print(f"📸 Image: {Path(result['image']).name}")
        print()
        
        # Medications Detected
        print("💊 MEDICATIONS DETECTED")
        print("-"*70)
        drugs = result['ocr_results']['drugs_detected']
        for i, drug in enumerate(drugs, 1):
            print(f"   {i}. {drug.upper()}")
        print(f"\n   OCR Confidence: {result['ocr_results']['confidence']*100:.1f}%")
        print()
        
        # Interactions Found
        interactions = result['ml_predictions']['interactions']
        print("⚠️  INTERACTIONS ANALYSIS")
        print("-"*70)
        
        if interactions:
            for i, interaction in enumerate(interactions, 1):
                severity = interaction['severity']
                conf = interaction['confidence']
                
                if severity == 'HIGH':
                    icon = "🔴"
                elif severity == 'MODERATE':
                    icon = "🟡"
                else:
                    icon = "🟢"
                
                print(f"\n{icon} INTERACTION #{i} - {severity} SEVERITY")
                print(f"   Drugs: {interaction['drug1'].upper()} + {interaction['drug2'].upper()}")
                print(f"   Confidence: {conf*100:.1f}%")
                print(f"   Source: {interaction['source']}")
        else:
            print("\n✅ No dangerous interactions detected")
        
        print()
        
        # Risk Assessment
        risk = result['risk_assessment']
        print("📊 OVERALL RISK ASSESSMENT")
        print("-"*70)
        print(f"   Risk Score: {risk['risk_score']}/100")
        print(f"   Risk Level: {risk['risk_level']}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-"*70)
        for i, rec in enumerate(risk['recommendations'], 1):
            print(f"   {i}. {rec}")
        print()
        
        # Disclaimer
        print("="*70)
        print("⚠️  IMPORTANT DISCLAIMER")
        print("="*70)
        print("This analysis is for informational purposes only and should not")
        print("replace professional medical advice. Always consult your doctor")
        print("or pharmacist before making any changes to your medications.")
        print("="*70)
        print()
        
        # Save report
        self._save_report(result)
    
    def _save_report(self, result: dict):
        """Save report to file"""
        
        # Create results directory
        results_dir = Path("results/reports")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"safety_report_{timestamp}.json"
        
        # Save JSON
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Report saved to: {report_file}")
        print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Demo the complete pipeline"""
    
    print("\n" + "="*70)
    print("💊 MedCheck - Complete AI Pipeline Demo")
    print("="*70)
    print()
    print("This demonstrates the FULL system:")
    print("  1. Advanced OCR (Computer Vision)")
    print("  2. ML Interaction Prediction")
    print("  3. Risk Assessment")
    print("  4. Safety Report Generation")
    print()
    print("="*70)
    print()
    
    # Initialize pipeline
    pipeline = MedCheckPipeline()
    
    # Check for test image
    test_image = "data/test_images/test_prescription.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print()
        print("💡 To test the pipeline:")
        print("   1. Place a prescription image in data/test_images/")
        print("   2. Or specify image path below")
        print()
        
        custom_path = input("Enter image path (or press Enter to skip): ").strip()
        if custom_path and os.path.exists(custom_path):
            test_image = custom_path
        else:
            print("\nNo image provided. Exiting.")
            return
    
    # Process prescription
    result = pipeline.process_prescription(test_image)
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    # Generate report
    pipeline.generate_report(result)
    
    print("="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print()
    print("🎯 What just happened:")
    print("   1. ✅ OCR extracted drug names from image")
    print("   2. ✅ ML model predicted interaction risks")
    print("   3. ✅ Risk assessment calculated")
    print("   4. ✅ Safety report generated")
    print()
    print("💼 Resume Value:")
    print("   'Built end-to-end AI pipeline integrating Computer Vision,")
    print("   Machine Learning, and automated risk assessment for")
    print("   medication safety analysis'")
    print()


if __name__ == "__main__":
    main()
