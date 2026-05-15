"""
Legitimate Drug Interaction Checker
Uses clinical standards: RxNorm (NIH) + FDA FAERS
"""

import requests
import time
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RxNormChecker:
    """
    RxNorm API - Clinical Standard (NIH/NLM)
    Used by: Epic, Cerner, all major EHR systems
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxNorm Concept Unique Identifier for a drug"""
        try:
            url = f"{self.BASE_URL}/rxcui.json"
            response = requests.get(url, params={'name': drug_name.strip()}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
                if rxcui_list:
                    logger.info(f"Found RxCUI for {drug_name}: {rxcui_list[0]}")
                    return rxcui_list[0]
            
            logger.warning(f"No RxCUI found for: {drug_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting RxCUI for {drug_name}: {e}")
            return None
    
    def get_interactions(self, rxcui: str) -> Dict:
        """Get all interactions for a drug from RxNorm"""
        try:
            url = f"{self.BASE_URL}/interaction/interaction.json"
            response = requests.get(
                url, 
                params={'rxcui': rxcui, 'sources': 'ONCHigh'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Retrieved interactions for RxCUI {rxcui}")
                return data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting interactions for RxCUI {rxcui}: {e}")
            return {}
    
    def check_pair(self, drug1: str, drug2: str) -> Dict:
        """Check if two drugs interact - CLINICAL STANDARD"""
        
        rxcui1 = self.get_rxcui(drug1)
        if not rxcui1:
            return {
                'found': False,
                'reason': f'Drug "{drug1}" not found in RxNorm database',
                'source': 'RxNorm (NIH/NLM)'
            }
        
        interactions_data = self.get_interactions(rxcui1)
        
        if interactions_data and 'interactionTypeGroup' in interactions_data:
            for group in interactions_data['interactionTypeGroup']:
                for int_type in group.get('interactionType', []):
                    for pair in int_type.get('interactionPair', []):
                        
                        interaction_concept = pair.get('interactionConcept', [])
                        if len(interaction_concept) > 1:
                            interacting_drug = interaction_concept[1].get('minConceptItem', {}).get('name', '').lower()
                            
                            if drug2.lower() in interacting_drug or interacting_drug in drug2.lower():
                                
                                description = pair.get('description', 'No description available')
                                severity = pair.get('severity', 'N/A')
                                
                                return {
                                    'found': True,
                                    'drug1': drug1,
                                    'drug2': drug2,
                                    'severity': severity.upper() if severity != 'N/A' else 'MODERATE',
                                    'description': description,
                                    'interacting_drug_name': interaction_concept[1].get('minConceptItem', {}).get('name', drug2),
                                    'source': 'RxNorm (NIH/NLM - Clinical Standard)',
                                    'source_url': 'https://www.nlm.nih.gov/research/umls/rxnorm/',
                                    'authority': 'HIGH'
                                }
        
        return {
            'found': False,
            'checked_against': 'RxNorm Clinical Database',
            'drug1': drug1,
            'drug2': drug2,
            'source': 'RxNorm (NIH/NLM)'
        }


class FDAFAERSChecker:
    """FDA FAERS - Real adverse event reports"""
    
    BASE_URL = "https://api.fda.gov/drug/event.json"
    
    def check_adverse_events(self, drug1: str, drug2: str) -> Dict:
        """Check FDA FAERS for real adverse events"""
        try:
            query = f'patient.drug.medicinalproduct:"{drug1}"+AND+patient.drug.medicinalproduct:"{drug2}"'
            
            response = requests.get(
                self.BASE_URL,
                params={
                    'search': query,
                    'count': 'patient.reaction.reactionmeddrapt.exact',
                    'limit': 10
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data:
                    reactions = data['results']
                    total_reports = sum(r['count'] for r in reactions)
                    
                    if total_reports >= 10:
                        
                        if total_reports > 100:
                            severity = 'HIGH'
                        elif total_reports > 50:
                            severity = 'MODERATE'
                        else:
                            severity = 'LOW'
                        
                        top_reactions = [f"{r['term']} ({r['count']} reports)" for r in reactions[:5]]
                        
                        return {
                            'found': True,
                            'drug1': drug1,
                            'drug2': drug2,
                            'total_reports': total_reports,
                            'severity': severity,
                            'top_reactions': top_reactions,
                            'description': f"Found {total_reports} adverse event reports involving both drugs.",
                            'source': 'FDA FAERS',
                            'authority': 'HIGH'
                        }
            
            return {'found': False}
            
        except Exception as e:
            logger.error(f"FDA FAERS Error: {e}")
            return {'found': False, 'error': str(e)}


class LegitimateInteractionChecker:
    """Enterprise-grade drug interaction checker"""
    
    def __init__(self):
        self.rxnorm = RxNormChecker()
        self.fda = FDAFAERSChecker()
    
    def check_interaction(self, drug1: str, drug2: str) -> Dict:
        """Check using clinical standards"""
        
        results = {
            'drug1': drug1,
            'drug2': drug2,
            'sources_checked': [],
            'interactions_found': []
        }
        
        # RxNorm
        rxnorm_result = self.rxnorm.check_pair(drug1, drug2)
        results['sources_checked'].append('RxNorm (NIH/NLM)')
        
        if rxnorm_result['found']:
            results['interactions_found'].append(rxnorm_result)
        
        # FDA FAERS
        fda_result = self.fda.check_adverse_events(drug1, drug2)
        results['sources_checked'].append('FDA FAERS')
        
        if fda_result['found']:
            results['interactions_found'].append(fda_result)
        
        # Verdict
        if results['interactions_found']:
            results['verdict'] = 'INTERACTION FOUND'
            severities = [i['severity'] for i in results['interactions_found']]
            results['overall_severity'] = 'HIGH' if 'HIGH' in severities else 'MODERATE'
            results['recommendation'] = '⚠️ Consult your doctor before taking these together.'
        else:
            results['verdict'] = 'NO KNOWN INTERACTION'
            results['overall_severity'] = 'NONE'
            results['recommendation'] = '✅ No major interactions found.'
        
        return results


def check_drug_interaction(drug1: str, drug2: str) -> Dict:
    """Quick check function"""
    checker = LegitimateInteractionChecker()
    return checker.check_interaction(drug1, drug2)
