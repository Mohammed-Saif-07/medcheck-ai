#!/usr/bin/env python3
"""
Process FDA FAERS data into drug interaction database
Extracts drug pairs with adverse events
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FDADataProcessor:
    """Process FDA FAERS adverse event data"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.drug_pairs = defaultdict(lambda: {
            'total_reports': 0,
            'serious_reports': 0,
            'reactions': defaultdict(int),
            'outcomes': defaultdict(int)
        })
    
    def process(self):
        """Process FDA FAERS JSON file"""
        
        logger.info(f"Processing {self.data_file}...")
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        logger.info(f"Found {len(results)} adverse event reports")
        
        processed = 0
        drug_pair_reports = 0
        
        for report in results:
            patient = report.get('patient', {})
            drugs = patient.get('drug', [])
            reactions = patient.get('reaction', [])
            
            if len(drugs) < 2:
                continue
            
            drug_names = []
            for drug in drugs:
                name = drug.get('medicinalproduct', '').strip().lower()
                if name and len(name) > 2:
                    drug_names.append(name)
            
            if len(drug_names) >= 2:
                for i in range(len(drug_names)):
                    for j in range(i + 1, len(drug_names)):
                        drug1, drug2 = sorted([drug_names[i], drug_names[j]])
                        pair_key = f"{drug1}|{drug2}"
                        
                        self.drug_pairs[pair_key]['total_reports'] += 1
                        
                        serious = report.get('serious', 0)
                        if serious:
                            self.drug_pairs[pair_key]['serious_reports'] += 1
                        
                        for reaction in reactions:
                            reaction_term = reaction.get('reactionmeddrapt', '').lower()
                            if reaction_term:
                                self.drug_pairs[pair_key]['reactions'][reaction_term] += 1
                        
                        outcome = report.get('seriousnessdeath', 0)
                        if outcome:
                            self.drug_pairs[pair_key]['outcomes']['death'] += 1
                        
                        drug_pair_reports += 1
            
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Processed {processed} reports")
        
        logger.info(f"✅ Complete! {len(self.drug_pairs)} unique pairs")
        return self.drug_pairs
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        
        records = []
        
        for pair_key, data in self.drug_pairs.items():
            drug1, drug2 = pair_key.split('|')
            
            top_reactions = sorted(
                data['reactions'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            total = data['total_reports']
            serious = data['serious_reports']
            severity_score = (serious / total) if total > 0 else 0
            
            if severity_score > 0.7 or total > 100:
                severity = 'HIGH'
            elif severity_score > 0.4 or total > 50:
                severity = 'MODERATE'
            else:
                severity = 'LOW'
            
            records.append({
                'drug1': drug1,
                'drug2': drug2,
                'total_reports': total,
                'serious_reports': serious,
                'severity_score': severity_score,
                'severity': severity,
                'top_reactions': ', '.join([r[0] for r in top_reactions]),
                'deaths': data['outcomes'].get('death', 0),
                'source': 'FDA FAERS'
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('total_reports', ascending=False)
        
        return df


def main():
    data_file = Path('data/fda_faers/drug-event-0001-of-0014.json')
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    processor = FDADataProcessor(data_file)
    drug_pairs = processor.process()
    
    df = processor.to_dataframe()
    
    output_file = Path('data/fda_interactions.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(df)} drug pairs to {output_file}")
    print(f"\nTop 10 interactions:")
    print(df[['drug1', 'drug2', 'total_reports', 'severity']].head(10))


if __name__ == '__main__':
    main()
