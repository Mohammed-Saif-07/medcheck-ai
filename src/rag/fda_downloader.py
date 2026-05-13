"""
MedCheck RAG - FDA OpenFDA Data Downloader

Downloads drug labels from FDA OpenFDA API
Builds knowledge base for RAG system

COMPLETELY FREE! No API key needed!
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class FDADownloader:
    """
    Download drug labels from FDA OpenFDA
    """
    
    def __init__(self, output_dir="data/raw/fda_labels"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.fda.gov/drug/label.json"
        
        # Common drugs to download (start with top 100)
        self.target_drugs = [
            'lisinopril', 'metformin', 'atorvastatin', 'amlodipine',
            'simvastatin', 'omeprazole', 'losartan', 'gabapentin',
            'warfarin', 'aspirin', 'ibuprofen', 'acetaminophen',
            'levothyroxine', 'metoprolol', 'hydrochlorothiazide',
            'albuterol', 'furosemide', 'pantoprazole', 'prednisone',
            'amoxicillin', 'azithromycin', 'clopidogrel', 'sertraline',
            'escitalopram', 'montelukast', 'rosuvastatin', 'esomeprazole',
            'ranitidine', 'tramadol', 'pravastatin', 'duloxetine',
            'valsartan', 'insulin', 'carvedilol', 'trazodone',
            'cyclobenzaprine', 'clonazepam', 'lorazepam', 'alprazolam',
            'diazepam', 'fluoxetine', 'paroxetine', 'venlafaxine',
            'bupropion', 'mirtazapine', 'quetiapine', 'aripiprazole',
            'olanzapine', 'risperidone', 'ziprasidone', 'lithium'
        ]
        
        print("📥 FDA OpenFDA Downloader Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target: {len(self.target_drugs)} common drugs")
        print()
    
    def download_drug_label(self, drug_name: str) -> dict:
        """Download label for a specific drug"""
        
        try:
            # Build query
            query = f"search=openfda.generic_name:{drug_name}&limit=1"
            url = f"{self.base_url}?{query}"
            
            # Make request
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0]
            
            # Try brand name if generic fails
            query = f"search=openfda.brand_name:{drug_name}&limit=1"
            url = f"{self.base_url}?{query}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0]
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error downloading {drug_name}: {e}")
            return None
    
    def extract_interactions_section(self, label: dict) -> str:
        """Extract drug interactions section from label"""
        
        sections = []
        
        # Drug interactions section
        if 'drug_interactions' in label:
            sections.append("DRUG INTERACTIONS:\n" + '\n'.join(label['drug_interactions']))
        
        # Warnings section (often contains interactions)
        if 'warnings' in label:
            sections.append("WARNINGS:\n" + '\n'.join(label['warnings']))
        
        # Contraindications
        if 'contraindications' in label:
            sections.append("CONTRAINDICATIONS:\n" + '\n'.join(label['contraindications']))
        
        # Adverse reactions (relevant for interactions)
        if 'adverse_reactions' in label:
            sections.append("ADVERSE REACTIONS:\n" + '\n'.join(label['adverse_reactions']))
        
        return '\n\n'.join(sections) if sections else None
    
    def download_all(self):
        """Download labels for all target drugs"""
        
        print("="*70)
        print("📥 DOWNLOADING FDA DRUG LABELS")
        print("="*70)
        print()
        
        labels_data = []
        successful = 0
        
        for i, drug in enumerate(tqdm(self.target_drugs, desc="Downloading"), 1):
            # Download label
            label = self.download_drug_label(drug)
            
            if label:
                # Extract key information
                drug_info = {
                    'drug_name': drug,
                    'generic_name': label.get('openfda', {}).get('generic_name', [drug])[0] if label.get('openfda', {}).get('generic_name') else drug,
                    'brand_names': label.get('openfda', {}).get('brand_name', []),
                    'drug_class': label.get('openfda', {}).get('pharm_class_epc', []),
                    'interactions_text': self.extract_interactions_section(label),
                    'full_label': label
                }
                
                labels_data.append(drug_info)
                successful += 1
                
                # Save individual label
                label_file = self.output_dir / f"{drug}_label.json"
                with open(label_file, 'w') as f:
                    json.dump(drug_info, f, indent=2)
            
            # Rate limiting (be nice to FDA API!)
            time.sleep(0.5)  # 2 requests per second max
        
        print()
        print("="*70)
        print(f"✅ Download Complete!")
        print(f"   Successfully downloaded: {successful}/{len(self.target_drugs)}")
        print(f"   Location: {self.output_dir.absolute()}")
        print("="*70)
        print()
        
        # Save combined data
        combined_file = self.output_dir.parent / "fda_labels_combined.json"
        with open(combined_file, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        print(f"💾 Combined data saved to: {combined_file}")
        print()
        
        return labels_data
    
    def quick_sample(self, n=10):
        """Quick download of just a few drugs for testing"""
        
        print("="*70)
        print("⚡ QUICK SAMPLE DOWNLOAD")
        print("="*70)
        print(f"\nDownloading {n} sample drugs for testing...\n")
        
        sample_drugs = self.target_drugs[:n]
        labels_data = []
        
        for drug in tqdm(sample_drugs, desc="Downloading"):
            label = self.download_drug_label(drug)
            
            if label:
                drug_info = {
                    'drug_name': drug,
                    'generic_name': label.get('openfda', {}).get('generic_name', [drug])[0] if label.get('openfda', {}).get('generic_name') else drug,
                    'brand_names': label.get('openfda', {}).get('brand_name', []),
                    'drug_class': label.get('openfda', {}).get('pharm_class_epc', []),
                    'interactions_text': self.extract_interactions_section(label),
                    'full_label': label
                }
                labels_data.append(drug_info)
                
                # Save individual label
                label_file = self.output_dir / f"{drug}_label.json"
                with open(label_file, 'w') as f:
                    json.dump(drug_info, f, indent=2)
            
            time.sleep(0.5)
        
        print(f"\n✅ Downloaded {len(labels_data)} sample labels!")
        
        # Save combined data
        combined_file = self.output_dir.parent / "fda_labels_combined.json"
        with open(combined_file, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        print(f"💾 Combined data saved to: {combined_file}")
        print()
        
        return labels_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main download workflow"""
    
    print("\n" + "="*70)
    print("💊 MedCheck RAG - FDA Data Downloader")
    print("="*70)
    print()
    print("This will download drug labels from FDA OpenFDA API")
    print()
    print("Options:")
    print("  1. Quick sample (10 drugs) - Recommended for testing!")
    print("  2. Common drugs (50 drugs) - Good balance")
    print("  3. Full download (All target drugs)")
    print()
    
    choice = input("Select option (1/2/3): ").strip()
    
    downloader = FDADownloader()
    
    if choice == "1":
        print("\n⚡ Quick Sample Selected")
        labels = downloader.quick_sample(10)
    elif choice == "2":
        print("\n📦 Common Drugs Selected")
        downloader.target_drugs = downloader.target_drugs[:50]
        labels = downloader.download_all()
    elif choice == "3":
        print("\n📦 Full Download Selected")
        labels = downloader.download_all()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("="*70)
    print("✅ FDA DATA DOWNLOAD COMPLETE!")
    print("="*70)
    print()
    print("📊 Summary:")
    print(f"   Total labels: {len(labels)}")
    print(f"   Storage location: data/raw/fda_labels/")
    print()
    print("🎯 Next step: Build vector database!")
    print("   Run: python src/rag/build_vector_db.py")
    print()


if __name__ == "__main__":
    main()
