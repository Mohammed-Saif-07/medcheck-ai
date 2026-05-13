"""
Medication Interaction Checker - IMPROVED VERSION v2.0
Fixed API issues + Better error handling + Debug mode

Author: Your Name
Date: May 2026
"""

import requests
import json
from typing import List, Dict
import time

class DrugChecker:
    """Improved drug interaction checker with better error handling"""
    
    def __init__(self, debug=True):
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        self.debug = debug
        
    def log(self, message):
        """Print debug messages"""
        if self.debug:
            print(message)
        
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
            
            # Debug: Show what we got
            self.log(f"   [DEBUG] Status Code: {response.status_code}")
            self.log(f"   [DEBUG] Response: {response.text[:200]}")
            
            # Check if response is valid
            if response.status_code != 200:
                print(f"   ❌ API Error: Status {response.status_code}")
                return None
            
            if not response.text.strip():
                print(f"   ❌ Empty response from API")
                return None
                
            data = response.json()
            
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                drug_id = data['idGroup']['rxnormId'][0]
                print(f"   ✅ Found ID: {drug_id}")
                return drug_id
            else:
                print(f"   ❌ Drug not found: {drug_name}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout - API too slow")
            return None
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection error - check internet")
            return None
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON response")
            self.log(f"   [DEBUG] Response was: {response.text[:500]}")
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def check_interactions_v2(self, drug_id: str, drug_name: str) -> List[Dict]:
        """
        IMPROVED: Check interactions using updated endpoint
        """
        # Try multiple sources
        sources = ['DrugBank', 'ONCHigh']
        
        for source in sources:
            url = f"{self.rxnorm_base}/interaction/interaction.json"
            params = {
                'rxcui': drug_id,
                'sources': source
            }
            
            try:
                self.log(f"   [DEBUG] Trying source: {source}")
                response = requests.get(url, params=params, timeout=15)
                
                # Debug output
                self.log(f"   [DEBUG] Status: {response.status_code}")
                self.log(f"   [DEBUG] Response length: {len(response.text)}")
                
                if response.status_code != 200:
                    continue
                
                if not response.text.strip():
                    self.log(f"   [DEBUG] Empty response, trying next source...")
                    continue
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
                data = response.json()
                
                interactions = []
                
                # Parse the response
                if 'interactionTypeGroup' in data:
                    for group in data['interactionTypeGroup']:
                        if 'interactionType' not in group:
                            continue
                            
                        for interaction_type in group['interactionType']:
                            if 'interactionPair' not in interaction_type:
                                continue
                                
                            for pair in interaction_type['interactionPair']:
                                try:
                                    interactions.append({
                                        'drug': pair['interactionConcept'][1]['minConceptItem']['name'],
                                        'description': pair['description'],
                                        'severity': pair.get('severity', 'Not specified')
                                    })
                                except (KeyError, IndexError):
                                    continue
                
                if interactions:
                    self.log(f"   [DEBUG] Found {len(interactions)} interactions from {source}")
                    return interactions
                else:
                    self.log(f"   [DEBUG] No interactions in {source}, trying next...")
                    
            except requests.exceptions.Timeout:
                self.log(f"   [DEBUG] Timeout on {source}")
                continue
            except json.JSONDecodeError:
                self.log(f"   [DEBUG] Invalid JSON from {source}")
                continue
            except Exception as e:
                self.log(f"   [DEBUG] Error with {source}: {e}")
                continue
        
        return []
    
    def analyze_medications(self, drug_names: List[str]):
        """
        Main function: analyze a list of medications
        """
        print("\n" + "="*60)
        print("💊 MEDICATION INTERACTION CHECKER v2.0")
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
            time.sleep(0.3)  # Small delay between requests
        
        print(f"\n✅ Successfully identified {len(drug_ids)}/{len(drug_names)} drugs\n")
        
        if len(drug_ids) == 0:
            print("❌ Could not identify any drugs. Check spelling or try common names.")
            return
        
        # Check interactions for each drug
        all_interactions = {}
        
        for drug, drug_id in drug_ids.items():
            print(f"\n🔬 Checking interactions for {drug.title()}...")
            interactions = self.check_interactions_v2(drug_id, drug)
            
            if interactions:
                # Filter to only show interactions with drugs in our list
                relevant_interactions = [
                    i for i in interactions 
                    if any(other_drug.lower() in i['drug'].lower() 
                          for other_drug in drug_names if other_drug != drug)
                ]
                
                if relevant_interactions:
                    all_interactions[drug] = relevant_interactions
                    print(f"   ⚠️  Found {len(relevant_interactions)} interactions with your medications!")
                else:
                    print(f"   ℹ️  Found {len(interactions)} total interactions, but none with your current medications")
                    print(f"   ✅ Safe with your other medications")
            else:
                print(f"   ✅ No known interactions found")
        
        # Display results
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60 + "\n")
        
        if all_interactions:
            print(f"⚠️  WARNING: Found {sum(len(v) for v in all_interactions.values())} interactions!\n")
            
            for drug, interactions in all_interactions.items():
                print(f"\n🔴 {drug.upper()}")
                print("-" * 60)
                
                for i, interaction in enumerate(interactions, 1):
                    print(f"\n   Interaction #{i}:")
                    print(f"   ⚠️  Interacts with: {interaction['drug']}")
                    print(f"   📊 Severity: {interaction['severity']}")
                    print(f"   📝 Details: {interaction['description'][:200]}...")
        else:
            print("✅ GOOD NEWS: No interactions found between your current medications!")
            print("\nNote: This checks interactions between the medications you listed.")
            print("Each drug may still have interactions with OTHER medications not in your list.")
        
        print("\n" + "="*60)
        print("⚠️  IMPORTANT DISCLAIMER:")
        print("   • This is for informational purposes only")
        print("   • Always consult your doctor or pharmacist")
        print("   • Do not stop or change medications without medical advice")
        print("="*60 + "\n")


# ============================================================================
# MAIN PROGRAM - EDIT THIS PART TO TEST YOUR MEDICATIONS
# ============================================================================

if __name__ == "__main__":
    # Create checker instance (debug=True shows detailed logs)
    checker = DrugChecker(debug=False)  # Set to True to see API details
    
    # ⬇️ EDIT THIS LIST - Add your medications here
    # Common examples that WILL show interactions:
    
    # Example 1: Known dangerous combination
    my_medications = [
        "warfarin",      # Blood thinner
        "aspirin",       # Also thins blood - DANGEROUS with warfarin!
        "ibuprofen"      # NSAID - also risky with warfarin
    ]
    
    # Example 2: Common diabetes + heart medications
    # my_medications = [
    #     "metformin",     # Diabetes
    #     "lisinopril",    # Blood pressure
    #     "atorvastatin"   # Cholesterol
    # ]
    
    # Example 3: Try your own medications
    # my_medications = [
    #     "your_medication_1",
    #     "your_medication_2",
    #     "your_medication_3"
    # ]
    
    # Run the analysis
    print("\n🚀 Starting analysis...\n")
    checker.analyze_medications(my_medications)
    
    print("\n" + "="*60)
    print("💡 TIPS:")
    print("   • Edit 'my_medications' list above to check different drugs")
    print("   • Use generic names (e.g., 'ibuprofen' not 'Advil')")
    print("   • Set debug=True in DrugChecker() to see API details")
    print("   • Try: metformin, lisinopril, atorvastatin, omeprazole, etc.")
    print("="*60 + "\n")
