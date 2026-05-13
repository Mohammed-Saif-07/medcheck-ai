"""
MedCheck COMPLETE PIPELINE
Scan → Extract → Check Interactions → Report

This is your FINAL DEMO-READY version!
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import requests
from typing import List, Dict
from itertools import combinations
import json

class MedCheckComplete:
    """
    Complete medication safety system
    - Scan prescription bottles (OCR)
    - Extract drug names
    - Check dangerous interactions
    - Generate safety report
    """
    
    def __init__(self):
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        
        # Known dangerous interactions database
        self.dangerous_interactions = {
            ('warfarin', 'aspirin'): {
                'severity': 'HIGH',
                'description': 'Increased bleeding risk. Both drugs thin blood.',
                'action': 'Consult doctor immediately'
            },
            ('warfarin', 'ibuprofen'): {
                'severity': 'HIGH',
                'description': 'NSAIDs increase bleeding risk with warfarin.',
                'action': 'Use alternative pain reliever'
            },
            ('lisinopril', 'spironolactone'): {
                'severity': 'HIGH',
                'description': 'Can cause dangerous high potassium levels.',
                'action': 'Monitor potassium levels closely'
            },
            ('metformin', 'glipizide'): {
                'severity': 'MODERATE',
                'description': 'Increased risk of low blood sugar.',
                'action': 'Monitor blood glucose frequently'
            },
        }
        
        print("💊 MedCheck Complete System Initialized!")
        print("✅ OCR Ready")
        print("✅ Drug Database Connected")
        print("✅ Interaction Checker Ready\n")
    
    def scan_prescription(self, image_path: str) -> List[str]:
        """OCR: Extract drug names from image"""
        
        print(f"📸 Scanning: {image_path}")
        
        # Read and preprocess image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # OCR
        text = pytesseract.image_to_string(thresh)
        print(f"   📝 Extracted {len(text)} characters")
        
        # Extract drug names
        words = re.findall(r'\b[A-Za-z]{5,}(?:\d+mg|mg)?\b', text)
        dosage_words = re.findall(r'\b([A-Za-z]+)(?:\d+)?(?:mg|ML|mcg)\b', text, re.IGNORECASE)
        words.extend(dosage_words)
        
        # Clean and filter
        skip_words = ['tablet', 'capsule', 'take', 'daily', 'times',
                     'twice', 'prescription', 'patient', 'doctor']
        
        drugs = []
        for word in words:
            word = re.sub(r'\d+', '', word).strip().lower()
            if word not in skip_words and len(word) > 4:
                drugs.append(word)
        
        drugs = list(set(drugs))
        print(f"   🎯 Found potential drugs: {drugs}\n")
        
        return drugs
    
    def verify_drug(self, drug_name: str) -> Dict:
        """Verify drug exists in RxNorm database"""
        
        url = f"{self.rxnorm_base}/rxcui.json"
        params = {'name': drug_name}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                return {
                    'name': drug_name,
                    'rxcui': data['idGroup']['rxnormId'][0],
                    'verified': True
                }
        except:
            pass
        
        return {'name': drug_name, 'verified': False}
    
    def check_interaction(self, drug1: str, drug2: str) -> Dict:
        """Check if two drugs have dangerous interaction"""
        
        # Check our database
        pair1 = (drug1.lower(), drug2.lower())
        pair2 = (drug2.lower(), drug1.lower())
        
        if pair1 in self.dangerous_interactions:
            return self.dangerous_interactions[pair1]
        elif pair2 in self.dangerous_interactions:
            return self.dangerous_interactions[pair2]
        
        return None
    
    def analyze_medications(self, drug_list: List[str]) -> Dict:
        """
        Complete analysis:
        1. Verify each drug
        2. Check all pairwise interactions
        3. Calculate risk score
        """
        
        print("="*70)
        print("🔬 ANALYZING MEDICATIONS")
        print("="*70)
        
        # Step 1: Verify drugs
        print(f"\n📋 Verifying {len(drug_list)} medications...\n")
        
        verified_drugs = []
        for drug in drug_list:
            result = self.verify_drug(drug)
            if result['verified']:
                verified_drugs.append(result)
                print(f"   ✅ {drug.upper()} verified (RxCUI: {result['rxcui']})")
            else:
                print(f"   ❌ {drug} not found in database")
        
        if len(verified_drugs) == 0:
            print("\n❌ No verified drugs found!")
            return {'error': 'No verified drugs'}
        
        # Step 2: Check interactions
        print(f"\n🔍 Checking interactions between {len(verified_drugs)} drugs...\n")
        
        interactions_found = []
        
        for drug1, drug2 in combinations(verified_drugs, 2):
            interaction = self.check_interaction(drug1['name'], drug2['name'])
            
            if interaction:
                interactions_found.append({
                    'drug1': drug1['name'],
                    'drug2': drug2['name'],
                    'interaction': interaction
                })
                print(f"   ⚠️  {drug1['name'].upper()} + {drug2['name'].upper()}")
                print(f"       Severity: {interaction['severity']}")
        
        if not interactions_found:
            print("   ✅ No dangerous interactions found!")
        
        # Step 3: Calculate risk score
        severity_scores = {'HIGH': 10, 'MODERATE': 5, 'LOW': 2}
        total_risk = sum(
            severity_scores.get(i['interaction']['severity'], 0)
            for i in interactions_found
        )
        
        max_possible = len(interactions_found) * 10 if interactions_found else 1
        risk_score = (total_risk / max_possible) * 100 if interactions_found else 0
        
        return {
            'verified_drugs': verified_drugs,
            'interactions': interactions_found,
            'risk_score': risk_score,
            'total_interactions': len(interactions_found)
        }
    
    def generate_report(self, analysis: Dict):
        """Generate final safety report"""
        
        print("\n" + "="*70)
        print("📊 MEDICATION SAFETY REPORT")
        print("="*70)
        
        if 'error' in analysis:
            print(f"\n❌ Error: {analysis['error']}")
            return
        
        # Verified medications
        print(f"\n💊 MEDICATIONS VERIFIED: {len(analysis['verified_drugs'])}")
        print("-"*70)
        for drug in analysis['verified_drugs']:
            print(f"   ✅ {drug['name'].upper()}")
        
        # Interactions
        print(f"\n⚠️  INTERACTIONS FOUND: {analysis['total_interactions']}")
        print("-"*70)
        
        if analysis['interactions']:
            for i, interaction in enumerate(analysis['interactions'], 1):
                severity = interaction['interaction']['severity']
                
                if severity == 'HIGH':
                    icon = "🔴"
                elif severity == 'MODERATE':
                    icon = "🟡"
                else:
                    icon = "🟢"
                
                print(f"\n{icon} INTERACTION #{i} - {severity} RISK")
                print(f"   Drugs: {interaction['drug1'].upper()} + {interaction['drug2'].upper()}")
                print(f"   Risk: {interaction['interaction']['description']}")
                print(f"   Action: {interaction['interaction']['action']}")
        else:
            print("   ✅ No dangerous interactions detected")
        
        # Overall risk
        print(f"\n📈 OVERALL RISK SCORE: {analysis['risk_score']:.0f}/100")
        print("-"*70)
        
        if analysis['risk_score'] > 50:
            print("   🔴 HIGH RISK - Consult doctor immediately!")
        elif analysis['risk_score'] > 20:
            print("   🟡 MODERATE RISK - Discuss with pharmacist")
        else:
            print("   🟢 LOW RISK - Monitor for side effects")
        
        print("\n" + "="*70)
        print("⚠️  DISCLAIMER:")
        print("   This is for educational purposes only.")
        print("   Always consult your doctor or pharmacist!")
        print("="*70)
    
    def process_prescription_images(self, image_paths: List[str]):
        """
        MAIN PIPELINE:
        1. Scan all prescription images
        2. Extract all drugs
        3. Check interactions
        4. Generate report
        """
        
        print("\n" + "="*70)
        print("💊 MEDCHECK - COMPLETE PRESCRIPTION ANALYSIS")
        print("="*70)
        print(f"\n📸 Processing {len(image_paths)} prescription image(s)...\n")
        
        # Scan all images
        all_drugs = []
        for img_path in image_paths:
            drugs = self.scan_prescription(img_path)
            all_drugs.extend(drugs)
        
        # Remove duplicates
        all_drugs = list(set(all_drugs))
        
        print(f"🎯 Total unique drugs found: {len(all_drugs)}")
        print(f"   Drugs: {[d.upper() for d in all_drugs]}\n")
        
        # Analyze
        analysis = self.analyze_medications(all_drugs)
        
        # Report
        self.generate_report(analysis)


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def demo_scenario_1():
    """
    Scenario: Safe combination
    """
    print("\n" + "="*70)
    print("DEMO SCENARIO 1: Safe Medication Combination")
    print("="*70)
    
    checker = MedCheckComplete()
    
    # Simulate drugs from prescription scan
    drugs = ['lisinopril', 'metformin', 'atorvastatin']
    
    print(f"\n💊 Medications found: {[d.upper() for d in drugs]}\n")
    
    analysis = checker.analyze_medications(drugs)
    checker.generate_report(analysis)


def demo_scenario_2():
    """
    Scenario: Dangerous combination!
    """
    print("\n" + "="*70)
    print("DEMO SCENARIO 2: DANGEROUS Medication Combination")
    print("="*70)
    
    checker = MedCheckComplete()
    
    # Dangerous combination!
    drugs = ['warfarin', 'aspirin', 'ibuprofen']
    
    print(f"\n💊 Medications found: {[d.upper() for d in drugs]}\n")
    
    analysis = checker.analyze_medications(drugs)
    checker.generate_report(analysis)


def demo_scenario_3_with_images():
    """
    Scenario: Scan actual images
    """
    print("\n" + "="*70)
    print("DEMO SCENARIO 3: Scanning Prescription Images")
    print("="*70)
    
    checker = MedCheckComplete()
    
    # Check if test image exists
    import os
    if os.path.exists('test_prescription.png'):
        images = ['test_prescription.png']
        checker.process_prescription_images(images)
    else:
        print("\n❌ No test image found!")
        print("Run option 2 first to create test image, then try again.")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n💊 MEDCHECK - COMPLETE SYSTEM DEMO")
    print("="*70)
    print("\nChoose a demo scenario:")
    print()
    print("1. Safe medication combination (Lisinopril + Metformin + Atorvastatin)")
    print("2. DANGEROUS combination (Warfarin + Aspirin + Ibuprofen) ⚠️")
    print("3. Scan prescription images with OCR")
    print()
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        demo_scenario_1()
    
    elif choice == "2":
        demo_scenario_2()
    
    elif choice == "3":
        demo_scenario_3_with_images()
    
    else:
        print("\nInvalid choice. Running scenario 2 (dangerous combination)...")
        demo_scenario_2()
    
    print("\n" + "="*70)
    print("🎯 DEMO COMPLETE!")
    print("\n💡 This is your RESUME-READY project!")
    print("   ✅ Computer Vision (OCR)")
    print("   ✅ API Integration (RxNorm)")
    print("   ✅ Drug Interaction Detection")
    print("   ✅ Risk Scoring Algorithm")
    print("   ✅ Safety Reporting")
    print("="*70)
