"""
FDA FAERS + PubMed Drug Interaction Checker
Uses REAL government data - same sources as clinical systems
Source: FDA Adverse Event Reporting System (FAERS)
        PubMed Clinical Literature
"""

import requests
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# TIER 1: FDA FAERS API (Real-world adverse events)
# ============================================================

class FDAFAERSChecker:
    """
    FDA Adverse Event Reporting System
    10M+ real patient adverse event reports
    Used by: FDA, pharmacovigilance teams, hospitals
    """
    
    BASE_URL = "https://api.fda.gov/drug/event.json"
    
    def check_interaction(self, drug1: str, drug2: str) -> Dict:
        """Check FDA FAERS for adverse events with both drugs"""
        
        query = f'patient.drug.medicinalproduct:"{drug1}"+AND+patient.drug.medicinalproduct:"{drug2}"'
        
        try:
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
                    
                    if total_reports >= 5:
                        
                        if total_reports > 100:
                            severity = 'HIGH'
                        elif total_reports > 30:
                            severity = 'MODERATE'
                        else:
                            severity = 'LOW'
                        
                        top_reactions = [
                            f"{r['term'].title()} ({r['count']} reports)"
                            for r in reactions[:5]
                        ]
                        
                        return {
                            'found': True,
                            'source': 'FDA FAERS (Adverse Event Reporting System)',
                            'authority': 'US Government - FDA',
                            'total_reports': total_reports,
                            'severity': severity,
                            'top_reactions': top_reactions,
                            'description': (
                                f"Found {total_reports:,} adverse event reports in FDA database "
                                f"involving both {drug1.title()} and {drug2.title()}. "
                                f"Most common adverse reactions: "
                                f"{', '.join([r['term'] for r in reactions[:3]])}."
                            ),
                            'source_url': 'https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers'
                        }
            
            # If 404, no reports found
            return {
                'found': False,
                'source': 'FDA FAERS',
                'checked': True
            }
            
        except Exception as e:
            logger.error(f"FDA FAERS error for {drug1}+{drug2}: {e}")
            return {'found': False, 'error': str(e)}
    
    def get_drug_profile(self, drug_name: str) -> Dict:
        """Get general adverse event profile for a drug"""
        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    'search': f'patient.drug.medicinalproduct:"{drug_name}"',
                    'count': 'patient.reaction.reactionmeddrapt.exact',
                    'limit': 10
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return {
                        'drug': drug_name,
                        'top_adverse_events': [
                            r['term'] for r in data['results'][:10]
                        ],
                        'source': 'FDA FAERS'
                    }
            
            return {'drug': drug_name, 'found': False}
            
        except Exception as e:
            logger.error(f"Error getting drug profile for {drug_name}: {e}")
            return {'drug': drug_name, 'error': str(e)}


# ============================================================
# TIER 2: PubMed Literature Checker
# ============================================================

class PubMedChecker:
    """
    PubMed Clinical Literature Search
    30M+ peer-reviewed medical papers
    Used by: Researchers, clinicians worldwide
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def search_interaction(self, drug1: str, drug2: str, max_results: int = 5) -> Dict:
        """Search PubMed for clinical papers about this drug interaction"""
        
        query = (
            f'("{drug1}"[Title/Abstract] AND "{drug2}"[Title/Abstract] '
            f'AND (interaction OR adverse OR toxicity OR contraindication))'
        )
        
        try:
            # Search for PMIDs
            search_response = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={
                    'db': 'pubmed',
                    'term': query,
                    'retmax': max_results,
                    'retmode': 'json'
                },
                timeout=10
            )
            
            if search_response.status_code != 200:
                return {'found': False, 'source': 'PubMed'}
            
            pmids = search_response.json().get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return {'found': False, 'source': 'PubMed'}
            
            # Fetch paper details
            time.sleep(0.3)  # Rate limiting
            
            fetch_response = requests.get(
                f"{self.BASE_URL}/esummary.fcgi",
                params={
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'json'
                },
                timeout=10
            )
            
            if fetch_response.status_code != 200:
                return {'found': False, 'source': 'PubMed'}
            
            result_data = fetch_response.json().get('result', {})
            
            papers = []
            for pmid in pmids:
                if pmid in result_data:
                    paper = result_data[pmid]
                    papers.append({
                        'pmid': pmid,
                        'title': paper.get('title', 'No title'),
                        'journal': paper.get('source', 'Unknown'),
                        'year': paper.get('pubdate', 'Unknown')[:4],
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
            
            if papers:
                return {
                    'found': True,
                    'source': 'PubMed (Clinical Literature)',
                    'authority': 'NIH National Library of Medicine',
                    'paper_count': len(papers),
                    'papers': papers,
                    'description': (
                        f"Found {len(papers)} peer-reviewed clinical papers about "
                        f"{drug1.title()} and {drug2.title()} interactions."
                    ),
                    'source_url': 'https://pubmed.ncbi.nlm.nih.gov/'
                }
            
            return {'found': False, 'source': 'PubMed'}
            
        except Exception as e:
            logger.error(f"PubMed error for {drug1}+{drug2}: {e}")
            return {'found': False, 'error': str(e)}


# ============================================================
# TIER 3: RxNorm Drug Information
# ============================================================

class RxNormChecker:
    """
    NIH RxNorm - Clinical drug naming standard
    Used by: Epic, Cerner, all major EHR systems
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """Get drug information from RxNorm"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/rxcui.json",
                params={'name': drug_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
                
                if rxcui_list:
                    return {
                        'found': True,
                        'drug': drug_name,
                        'rxcui': rxcui_list[0],
                        'source': 'RxNorm (NIH)'
                    }
            
            return {'found': False, 'drug': drug_name}
            
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def get_drug_class(self, rxcui: str) -> List[str]:
        """Get drug class/category from RxNorm"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/rxclass/class/byRxcui.json",
                params={'rxcui': rxcui, 'relaSource': 'ATC'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                classes = []
                for concept in data.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', []):
                    class_name = concept.get('rxclassMinConceptItem', {}).get('className', '')
                    if class_name:
                        classes.append(class_name)
                return classes[:3]
            
            return []
            
        except Exception:
            return []


# ============================================================
# MAIN: Legitimate Drug Interaction Checker
# ============================================================

class LegitimateInteractionChecker:
    """
    Enterprise-grade drug interaction checker
    Using real clinical data sources:
    - FDA FAERS (10M+ real adverse events)
    - PubMed (30M+ clinical papers)
    - RxNorm (NIH drug standard)
    
    Same sources used by major healthcare companies!
    """
    
    def __init__(self):
        self.fda = FDAFAERSChecker()
        self.pubmed = PubMedChecker()
        self.rxnorm = RxNormChecker()
        logger.info("✅ Initialized with FDA FAERS + PubMed + RxNorm")
    
    def check_interaction(self, drug1: str, drug2: str) -> Dict:
        """
        Full interaction check using all clinical sources
        """
        
        drug1 = drug1.strip().lower()
        drug2 = drug2.strip().lower()
        
        logger.info(f"Checking interaction: {drug1} + {drug2}")
        
        results = {
            'drug1': drug1.title(),
            'drug2': drug2.title(),
            'sources_checked': [],
            'interactions_found': [],
            'clinical_papers': [],
            'drug1_info': {},
            'drug2_info': {}
        }
        
        # Get drug info from RxNorm
        results['drug1_info'] = self.rxnorm.get_drug_info(drug1)
        results['drug2_info'] = self.rxnorm.get_drug_info(drug2)
        
        # TIER 1: FDA FAERS
        logger.info("Checking FDA FAERS...")
        fda_result = self.fda.check_interaction(drug1, drug2)
        results['sources_checked'].append('FDA FAERS')
        
        if fda_result.get('found'):
            results['interactions_found'].append(fda_result)
        
        # TIER 2: PubMed
        logger.info("Searching PubMed...")
        pubmed_result = self.pubmed.search_interaction(drug1, drug2)
        results['sources_checked'].append('PubMed')
        
        if pubmed_result.get('found'):
            results['clinical_papers'] = pubmed_result.get('papers', [])
            results['interactions_found'].append(pubmed_result)
        
        # Final verdict
        if results['interactions_found']:
            results['verdict'] = 'INTERACTION FOUND'
            
            severities = [
                i.get('severity', 'LOW')
                for i in results['interactions_found']
                if 'severity' in i
            ]
            
            if 'HIGH' in severities:
                results['overall_severity'] = 'HIGH'
            elif 'MODERATE' in severities:
                results['overall_severity'] = 'MODERATE'
            elif severities:
                results['overall_severity'] = 'LOW'
            else:
                # PubMed found papers = at least moderate concern
                results['overall_severity'] = 'MODERATE'
            
            results['recommendation'] = (
                f"⚠️ IMPORTANT: Clinical data suggests a potential interaction between "
                f"{drug1.title()} and {drug2.title()}. "
                f"Contact your doctor or pharmacist before taking these together."
            )
        else:
            results['verdict'] = 'NO KNOWN INTERACTION'
            results['overall_severity'] = 'NONE'
            results['recommendation'] = (
                f"✅ No significant adverse event reports found in FDA database "
                f"for {drug1.title()} + {drug2.title()} combination. "
                f"Always inform your healthcare provider about all medications."
            )
        
        return results
    
    def check_multiple(self, drugs: List[str]) -> Dict:
        """Check all drug pairs in a list"""
        
        results = {
            'drugs': [d.title() for d in drugs],
            'pairs_checked': 0,
            'interactions': [],
            'sources': ['FDA FAERS', 'PubMed', 'RxNorm']
        }
        
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                result = self.check_interaction(drugs[i], drugs[j])
                results['pairs_checked'] += 1
                
                if result['verdict'] == 'INTERACTION FOUND':
                    results['interactions'].append(result)
        
        return results


# ============================================================
# Test Script
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LEGITIMATE DRUG INTERACTION CHECKER")
    print("Sources: FDA FAERS + PubMed + RxNorm")
    print("=" * 60)
    
    checker = LegitimateInteractionChecker()
    
    # Test 1: Known interaction
    print("\n🧪 Test 1: Warfarin + Aspirin")
    result = checker.check_interaction('warfarin', 'aspirin')
    print(f"Verdict: {result['verdict']}")
    print(f"Severity: {result.get('overall_severity')}")
    print(f"Sources: {result['sources_checked']}")
    
    if result['interactions_found']:
        for interaction in result['interactions_found']:
            print(f"\n  📊 {interaction['source']}:")
            print(f"     {interaction.get('description', '')[:100]}...")
    
    if result['clinical_papers']:
        print(f"\n  📚 {len(result['clinical_papers'])} clinical papers found:")
        for paper in result['clinical_papers'][:2]:
            print(f"     - {paper['title'][:70]}...")
            print(f"       {paper['url']}")
    
    print("\n" + "=" * 60)
    
    # Test 2: Safe combination
    print("\n🧪 Test 2: Lisinopril + Metformin")
    result = checker.check_interaction('lisinopril', 'metformin')
    print(f"Verdict: {result['verdict']}")
    print(f"Sources: {result['sources_checked']}")
    print(f"Recommendation: {result['recommendation'][:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE!")
