"""
Medication Interaction Checker - STARTER VERSION
Simple script to check drug interactions using RxNorm API (FREE, no signup!)

Author: Your Name
Date: May 2026
"""

import requests
import json
from typing import List, Dict

class DrugChecker:
    """Simple drug interaction checker using RxNorm API"""
    
    def __init__(self):
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        
    def get_drug_id(self, drug_name: str) -> str:
        """
        Get RxNorm ID for a drug name
        Example: "aspirin" -> "1191"
        """
        print(f"🔍 Looking up: {drug_name}...")
        
        url = f"{self.rxnorm_base}/rxcui.json"
        params = {'name': drug_name}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                drug_id = data['idGroup']['rxnormId'][0]
                print(f"   ✅ Found ID: {drug_id}")
                return drug_id
            else:
                print(f"   ❌ Drug not found: {drug_name}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def check_interactions(self, drug_id: str) -> List[Dict]:
        """
        Check what drugs interact with this drug
        Returns list of interactions
        """
        url = f"{self.rxnorm_base}/interaction/interaction.json"
        params = {'rxcui': drug_id}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            interactions = []
            
            if 'interactionTypeGroup' in data:
                for group in data['interactionTypeGroup']:
                    if 'interactionType' not in group:
                        continue
                        
                    for interaction_type in group['interactionType']:
                        if 'interactionPair' not in interaction_type:
                            continue
                            
                        for pair in interaction_type['interactionPair']:
                            interactions.append({
                                'drug': pair['interactionConcept'][1]['minConceptItem']['name'],
                                'description': pair['description'],
                                'severity': pair.get('severity', 'Not specified')
                            })
            
            return interactions
            
        except Exception as e:
            print(f"❌ Error checking interactions: {e}")
            return []
    
    def analyze_medications(self, drug_names: List[str]):
        """
        Main function: analyze a list of medications
        """
        print("\n" + "="*60)
        print("💊 MEDICATION INTERACTION CHECKER")
        print("="*60 + "\n")
        
        print(f"📋 Analyzing {len(drug_names)} medications:")
        for i, drug in enumerate(drug_names, 1):
            print(f"   {i}. {drug.title()}")
        print()
        
        # Get IDs for all drugs
        drug_ids = {}
        for drug in drug_names:
            drug_id = self.get_drug_id(drug)
            if drug_id:
                drug_ids[drug] = drug_id
        
        print(f"\n✅ Successfully identified {len(drug_ids)}/{len(drug_names)} drugs\n")
        
        # Check interactions for each drug
        all_interactions = {}
        
        for drug, drug_id in drug_ids.items():
            print(f"\n🔬 Checking interactions for {drug.title()}...")
            interactions = self.check_interactions(drug_id)
            
            if interactions:
                # Filter to only show interactions with drugs in our list
                relevant_interactions = [
                    i for i in interactions 
                    if any(other_drug.lower() in i['drug'].lower() 
                          for other_drug in drug_names if other_drug != drug)
                ]
                
                if relevant_interactions:
                    all_interactions[drug] = relevant_interactions
                    print(f"   ⚠️  Found {len(relevant_interactions)} interactions!")
                else:
                    print(f"   ✅ No interactions with your other medications")
            else:
                print(f"   ✅ No known interactions")
        
        # Display results
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60 + "\n")
        
        if all_interactions:
            print(f"⚠️  WARNING: Found {len(all_interactions)} medications with interactions!\n")
            
            for drug, interactions in all_interactions.items():
                print(f"\n🔴 {drug.upper()}")
                print("-" * 60)
                
                for i, interaction in enumerate(interactions, 1):
                    print(f"\n   Interaction #{i}:")
                    print(f"   Interacts with: {interaction['drug']}")
                    print(f"   Severity: {interaction['severity']}")
                    print(f"   Details: {interaction['description']}")
        else:
            print("✅ GOOD NEWS: No interactions found between your medications!")
        
        print("\n" + "="*60)
        print("⚠️  DISCLAIMER: This is for informational purposes only.")
        print("Always consult your doctor or pharmacist!")
        print("="*60 + "\n")


# ============================================================================
# MAIN PROGRAM - EDIT THIS PART TO TEST YOUR MEDICATIONS
# ============================================================================

if __name__ == "__main__":
    # Create checker instance
    checker = DrugChecker()
    
    # ⬇️ EDIT THIS LIST - Add your medications here
    my_medications = [
        "warfarin",      # Blood thinner
        "aspirin",       # Pain reliever (also blood thinner!)
        "ibuprofen"      # Pain reliever (NSAID)
    ]
    
    # Run the analysis
    checker.analyze_medications(my_medications)
    
    print("\n💡 TIP: Edit the 'my_medications' list to check different drugs!")
    print("Example drugs to try: lisinopril, metformin, atorvastatin, omeprazole\n")
