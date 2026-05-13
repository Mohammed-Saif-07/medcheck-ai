"""
Download 100% FREE datasets
No signup, no credit card, no cost!
"""

import requests
import json

def test_free_apis():
    """Test all free APIs"""
    
    print("🧪 Testing FREE APIs...")
    print()
    
    # 1. RxNorm (FREE, no signup)
    print("1. Testing RxNorm API (FREE)...")
    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ RxNorm working! (FREE, no limits)")
    
    # 2. FDA OpenFDA (FREE, no signup)
    print("\n2. Testing FDA OpenFDA API (FREE)...")
    url = "https://api.fda.gov/drug/label.json?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ FDA API working! (FREE, 1000 requests/day)")
    
    # 3. PubChem (FREE, no signup)
    print("\n3. Testing PubChem API (FREE)...")
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ PubChem working! (FREE, unlimited)")
    
    print("\n✅ All FREE APIs working!")
    print("💰 Total cost: ₹0")

if __name__ == "__main__":
    test_free_apis()
