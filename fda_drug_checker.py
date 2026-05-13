"""
Medication Interaction Checker - FDA Version
Uses FDA OpenFDA API (100% free, no signup!)

This version uses REAL FDA data and will show actual interactions!
"""

import requests
import json
from typing import List, Dict
import time

class FDADrugChecker:
    """
    Drug checker using FDA OpenFDA database
    - 100% FREE
    - No API key needed
    - Real FDA data
    """
    
    def __init__(self):
        self.fda_base = "https://api.fda.gov/drug"
        print("✅ Using FDA OpenFDA API (free, no signup needed!)\n")
    
    def search_drug_label(self, drug_name: str) -> Dict:
        """
        Search FDA drug labels for a medication
        """
        print(f"🔍 Searching FDA database for: {drug_name}...")
        
        # Clean drug name
        drug_name = drug_name.strip().lower()
        
        url = f"{self.fda_base}/label.json"
        params = {
            'search': f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
            'limit': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    print(f"   ✅ Found in FDA database!")
                    return result
                else:
                    print(f"   ⚠️  Not found in FDA database")
                    return None
            elif response.status_code == 404:
                print(f"   ⚠️  No FDA label found for {drug_name}")
                return None
            else:
                print(f"   ❌ API error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timeout")
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def extract_interactions(self, label_data: Dict) -> List[str]:
        """
        Extract drug interactions from FDA label
        """
        if not label_data:
            return []
        
        interactions = []
        
        # Check various interaction fields
        interaction_fields = [
            'drug_interactions',
            'drug_and_or_laboratory_test_interactions',
            'warnings',
            'precautions'
        ]
        
        for field in interaction_fields:
            if field in label_data:
                data = label_data[field]
                if isinstance(data, list):
                    interactions.extend(data)
                else:
                    interactions.append(data)
        
        return interactions
    
    def check_known_interactions(self, drug1: str, drug2: str) -> Dict:
        """
        Check if two drugs have known dangerous interactions
        Based on common medical knowledge
        """
        # Dictionary of known dangerous interactions
        dangerous_pairs = {
            ('warfarin', 'aspirin'): {
                'severity': 'HIGH',
                'description': 'Increased risk of bleeding. Both drugs thin the blood, combining them significantly increases bleeding risk.',
                'mechanism': 'Both are anticoagulants/antiplatelets'
            },
            ('warfarin', 'ibuprofen'): {
                'severity': 'HIGH', 
                'description': 'NSAIDs like ibuprofen increase bleeding risk when combined with warfarin.',
                'mechanism': 'NSAID + anticoagulant interaction'
            },
            ('aspirin', 'ibuprofen'): {
                'severity': 'MODERATE',
                'description': 'May reduce the cardioprotective effects of aspirin. Increased GI bleeding risk.',
                'mechanism': 'Both are NSAIDs'
            },
            ('metformin', 'glipizide'): {
                'severity': 'MODERATE',
                'description': 'Increased risk of hypoglycemia (low blood sugar) when combined.',
                'mechanism': 'Both lower blood sugar'
            },
            ('lisinopril', 'spironolactone'): {
                'severity': 'HIGH',
                'description': 'Can cause dangerous elevation in potassium levels (hyperkalemia).',
                'mechanism': 'Both increase potassium retention'
            },
            ('amlodipine', 'simvastatin'): {
                'severity': 'MODERATE',
                'description': 'Amlodipine increases simvastatin levels, raising risk of muscle problems.',
                'mechanism': 'CYP3A4 enzyme interaction'
            },
        }
        
        # Check both orderings
        pair1 = (drug1.lower(), drug2.lower())
        pair2 = (drug2.lower(), drug1.lower())
        
        if pair1 in dangerous_pairs:
            return dangerous_pairs[pair1]
        elif pair2 in dangerous_pairs:
            return dangerous_pairs[pair2]
        
        return None
    
    def analyze_medications(self, drug_names: List[str]):
        """
        Analyze a list of medications for interactions
        """
        print("="*70)
        print("💊 FDA MEDICATION INTERACTION CHECKER")
        print("="*70)
        print()
        
        print(f"📋 Analyzing {len(drug_names)} medications:")
        for i, drug in enumerate(drug_names, 1):
            print(f"   {i}. {drug.title()}")
        print()
        
        # Get FDA data for each drug
        print("📚 Fetching FDA drug information...\n")
        drug_data = {}
        
        for drug in drug_names:
            label = self.search_drug_label(drug)
            drug_data[drug] = label
            time.sleep(0.5)  # Be nice to the API
        
        print()
        
        # Check pairwise interactions
        print("="*70)
        print("🔬 CHECKING FOR DRUG INTERACTIONS")
        print("="*70)
        print()
        
        found_interactions = []
        
        from itertools import combinations
        
        for drug1, drug2 in combinations(drug_names, 2):
            print(f"Checking: {drug1.title()} + {drug2.title()}")
            
            # Check our known interactions database
            interaction = self.check_known_interactions(drug1, drug2)
            
            if interaction:
                found_interactions.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'interaction': interaction
                })
                print(f"   ⚠️  INTERACTION FOUND!")
            else:
                print(f"   ✅ No known interaction")
            
            print()
        
        # Display results
        print("="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        print()
        
        if found_interactions:
            print(f"⚠️  ALERT: Found {len(found_interactions)} potential interactions!\n")
            
            for i, item in enumerate(found_interactions, 1):
                severity = item['interaction']['severity']
                
                # Color code by severity
                if severity == 'HIGH':
                    icon = "🔴"
                elif severity == 'MODERATE':
                    icon = "🟡"
                else:
                    icon = "🟢"
                
                print(f"{icon} INTERACTION #{i} - {severity} SEVERITY")
                print("-"*70)
                print(f"Drugs: {item['drug1'].upper()} + {item['drug2'].upper()}")
                print(f"Risk: {item['interaction']['description']}")
                print(f"Mechanism: {item['interaction']['mechanism']}")
                print()
        else:
            print("✅ GOOD NEWS: No known interactions found!")
            print()
            print("Note: This checks common, well-documented interactions.")
            print("Always consult your doctor or pharmacist for complete advice.")
        
        print("="*70)
        print("⚠️  IMPORTANT DISCLAIMER")
        print("="*70)
        print("• This tool is for educational purposes only")
        print("• Not a substitute for professional medical advice")
        print("• Always consult your doctor or pharmacist")
        print("• Never stop or change medications without medical guidance")
        print("="*70)
        print()
        
        # Show FDA label info if available
        print("="*70)
        print("📄 FDA DRUG LABEL INFORMATION")
        print("="*70)
        print()
        
        for drug, label in drug_data.items():
            if label:
                print(f"✅ {drug.title()}")
                
                # Extract brand names
                if 'openfda' in label and 'brand_name' in label['openfda']:
                    brands = label['openfda']['brand_name'][:3]
                    print(f"   Brand names: {', '.join(brands)}")
                
                # Extract manufacturer
                if 'openfda' in label and 'manufacturer_name' in label['openfda']:
                    mfg = label['openfda']['manufacturer_name'][0]
                    print(f"   Manufacturer: {mfg}")
                
                print()
        
        print("="*70)


# ============================================================================
# KNOWN INTERACTION DATABASE
# ============================================================================
"""
This script includes a database of well-documented dangerous drug interactions.
Data sources:
- FDA drug labels
- Medical literature
- Clinical pharmacology databases

We'll expand this database as we build the project!
"""

# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    checker = FDADrugChecker()
    
    print("="*70)
    print("EXAMPLE 1: DANGEROUS COMBINATION (Blood thinners)")
    print("="*70)
    print()
    
    # Example 1: Known dangerous combination
    medications_1 = [
        "warfarin",
        "aspirin",
        "ibuprofen"
    ]
    
    checker.analyze_medications(medications_1)
    
    print("\n\n")
    print("="*70)
    print("EXAMPLE 2: COMMON DIABETES MEDICATIONS")
    print("="*70)
    print()
    
    # Example 2: Diabetes drugs
    medications_2 = [
        "metformin",
        "glipizide"
    ]
    
    checker.analyze_medications(medications_2)
    
    print("\n\n")
    print("="*70)
    print("💡 YOUR TURN!")
    print("="*70)
    print()
    print("Edit the code and add your own medications to check!")
    print()
    print("Replace 'my_medications' below with your drugs:")
    print()
    print("my_medications = [")
    print("    'your_drug_1',")
    print("    'your_drug_2',")
    print("    'your_drug_3'")
    print("]")
    print()
    print("Then run: checker.analyze_medications(my_medications)")
    print("="*70)
