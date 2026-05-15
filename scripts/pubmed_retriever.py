#!/usr/bin/env python3
"""PubMed Drug Interaction Literature Retriever"""

import requests
import time
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedRetriever:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email="research@example.com"):
        self.email = email
        self.session = requests.Session()
    
    def search(self, drug1, drug2, max_results=10):
        query = f'("{drug1}"[Title/Abstract] AND "{drug2}"[Title/Abstract] AND (interaction OR adverse))'
        
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pmids = data.get('esearchresult', {}).get('idlist', [])
                return pmids
            return []
        except Exception as e:
            logger.error(f"Error: {e}")
            return []
    
    def fetch_abstracts(self, pmids):
        if not pmids:
            return []
        
        pmid_str = ','.join(pmids)
        fetch_url = f"{self.BASE_URL}/esummary.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid_str,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            time.sleep(0.5)
            response = self.session.get(fetch_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                
                papers = []
                for pmid in pmids:
                    if pmid in result:
                        paper_data = result[pmid]
                        papers.append({
                            'pmid': pmid,
                            'title': paper_data.get('title', ''),
                            'journal': paper_data.get('source', ''),
                            'pubdate': paper_data.get('pubdate', ''),
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
                return papers
            return []
        except Exception as e:
            logger.error(f"Error: {e}")
            return []
    
    def get_interaction_papers(self, drug1, drug2, max_results=5):
        pmids = self.search(drug1, drug2, max_results)
        if pmids:
            return self.fetch_abstracts(pmids)
        return []


def main():
    print("Testing PubMed Retrieval\n")
    
    retriever = PubMedRetriever()
    
    print("Test: Warfarin + Aspirin")
    papers = retriever.get_interaction_papers('warfarin', 'aspirin', max_results=5)
    
    if papers:
        print(f"✅ Found {len(papers)} papers:\n")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
            print(f"   {paper['journal']} ({paper['pubdate']})")
            print(f"   {paper['url']}\n")
    else:
        print("❌ No papers found")


if __name__ == '__main__':
    main()
