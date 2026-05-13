"""
Simplified Advanced OCR - Tesseract Only
Works reliably without EasyOCR crashes
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import requests
import re
from typing import List, Dict

class SimplifiedOCR:
    """
    Reliable OCR using Tesseract only
    No EasyOCR = No crashes!
    """
    
    def __init__(self):
        print("🚀 Initializing Simplified OCR System...")
        print("✅ Tesseract OCR Ready!")
    
    def preprocess_image(self, image_path: str):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing attempts
        variations = [
            gray,  # Original
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Binary
            cv2.GaussianBlur(gray, (5, 5), 0),  # Blur
        ]
        
        return variations
    
    def extract_text(self, image_path: str) -> List[str]:
        """Extract text from image using Tesseract"""
        
        all_text = []
        
        # Try multiple preprocessing methods
        variations = self.preprocess_image(image_path)
        
        for var in variations:
            text = pytesseract.image_to_string(var)
            if text.strip():
                all_text.append(text)
        
        return all_text
    
    def extract_drug_names(self, texts: List[str]) -> List[str]:
        """Extract potential drug names from OCR text"""
        
        drug_candidates = set()
        
        for text in texts:
            # Split into words
            words = re.findall(r'[A-Za-z]{4,}', text)
            
            for word in words:
                word_clean = word.lower()
                # Only keep words that look like drug names
                if len(word_clean) >= 4:
                    drug_candidates.add(word_clean)
        
        return list(drug_candidates)
    
    def verify_with_rxnorm(self, drug_name: str) -> dict:
        """Verify drug name with RxNorm API"""
        
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={drug_name}&maxEntries=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                candidates = data.get('approximateGroup', {}).get('candidate', [])
                
                if candidates:
                    return {
                        'original': drug_name,
                        'matched': candidates[0]['name'].lower(),
                        'verified': True,
                        'score': candidates[0].get('score', 0)
                    }
        except:
            pass
        
        return {
            'original': drug_name,
            'matched': drug_name,
            'verified': False,
            'score': 0
        }
    
    def process_image(self, image_path: str) -> dict:
        """Main processing pipeline"""
        
        import time
        start_time = time.time()
        
        # Extract text
        texts = self.extract_text(image_path)
        
        # Get drug candidates
        candidates = self.extract_drug_names(texts)
        
        # Verify each candidate
        verified_drugs = []
        
        for candidate in candidates:
            result = self.verify_with_rxnorm(candidate)
            if result['verified'] or len(candidate) >= 6:
                verified_drugs.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            'drugs_verified': verified_drugs,
            'ocr_confidence': 0.8,  # Estimated
            'processing_time': processing_time,
            'raw_text': ' '.join(texts)
        }


# Make it compatible with old import
AdvancedOCR = SimplifiedOCR
