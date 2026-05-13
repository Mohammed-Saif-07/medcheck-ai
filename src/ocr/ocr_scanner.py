"""
MedCheck OCR - Prescription Bottle Scanner
Extracts drug names from prescription bottle images

Features:
- Image preprocessing (OpenCV)
- OCR text extraction (Tesseract)
- Drug name detection
- Fuzzy matching with drug database
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import requests
from typing import List, Dict, Tuple

class PrescriptionOCR:
    """Extract drug names from prescription bottle images"""
    
    def __init__(self):
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        print("✅ OCR System initialized!")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Enhance image for better OCR
        Steps: grayscale → denoise → threshold → sharpen
        """
        print(f"\n📸 Processing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print("   🔄 Step 1: Converting to grayscale...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print("   🔄 Step 2: Denoising...")
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        print("   🔄 Step 3: Adaptive thresholding...")
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        print("   🔄 Step 4: Sharpening...")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)
        
        # Save preprocessed image for debugging
        debug_path = image_path.replace('.', '_processed.')
        cv2.imwrite(debug_path, sharpened)
        print(f"   💾 Saved processed image: {debug_path}")
        
        return sharpened
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text using Tesseract OCR
        """
        print("\n   🔍 Running OCR (Tesseract)...")
        
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text
        text = pytesseract.image_to_string(image, config=custom_config)
        
        print(f"   📝 Extracted text ({len(text)} characters)")
        
        return text
    
    def extract_potential_drug_names(self, text: str) -> List[str]:
        """
        Extract words that could be drug names
        Improved to handle ALL CAPS, mixed case, and numbers
        """
        print("\n   🔬 Analyzing text for drug names...")
        
        # Clean text
        text = text.replace('\n', ' ')
        text = text.replace('"', ' ')
        
        # Find words - now handles ALL CAPS, Title Case, and mixed
        # Pattern: word of 5+ letters (may have numbers attached)
        words = re.findall(r'\b[A-Za-z]{5,}(?:\d+mg|mg)?\b', text)
        
        # Also try to extract words before "mg" or dosages
        dosage_words = re.findall(r'\b([A-Za-z]+)(?:\d+)?(?:mg|ML|mcg)\b', text, re.IGNORECASE)
        words.extend(dosage_words)
        
        # Filter potential drug names
        potential_drugs = []
        for word in words:
            # Clean the word
            word = re.sub(r'\d+', '', word)  # Remove numbers
            word = word.strip()
            
            # Skip common non-drug words
            skip_words = ['tablet', 'capsule', 'take', 'daily', 'times', 
                         'twice', 'once', 'the', 'this', 'that', 'drug', 
                         'medication', 'doctor', 'pharmacy', 'refill', 
                         'date', 'patient', 'name', 'prescription',
                         'tabletaly', 'taka']  # Common OCR errors
            
            word_lower = word.lower()
            
            if word_lower not in skip_words and len(word) > 4:
                potential_drugs.append(word_lower)
        
        # Remove duplicates
        potential_drugs = list(set(potential_drugs))
        
        print(f"   📋 Found {len(potential_drugs)} potential drug names: {potential_drugs}")
        
        return potential_drugs
    
    def verify_drug_name(self, drug_name: str) -> Dict:
        """
        Verify if a word is a real drug using RxNorm API
        """
        url = f"{self.rxnorm_base}/rxcui.json"
        params = {'name': drug_name}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                rxcui = data['idGroup']['rxnormId'][0]
                return {
                    'found': True,
                    'name': drug_name,
                    'rxcui': rxcui,
                    'confidence': 'high'
                }
        except:
            pass
        
        return {'found': False, 'name': drug_name}
    
    def fuzzy_match_drug(self, word: str) -> Dict:
        """
        Try fuzzy matching for misspelled drug names
        """
        # Try with slight variations
        variations = [
            word.lower(),
            word.capitalize(),
            word.upper(),
            word.replace('1', 'i'),  # Common OCR errors
            word.replace('0', 'o'),
            word.replace('5', 's'),
        ]
        
        for variant in variations:
            result = self.verify_drug_name(variant)
            if result['found']:
                return result
        
        return {'found': False, 'name': word}
    
    def scan_prescription(self, image_path: str) -> Dict:
        """
        Main function: Scan prescription and extract drug names
        
        Returns:
            {
                'image': path,
                'text': extracted text,
                'drugs_found': [...],
                'confidence': score
            }
        """
        print("\n" + "="*60)
        print("💊 PRESCRIPTION SCANNER")
        print("="*60)
        
        try:
            # Step 1: Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Step 2: Extract text
            text = self.extract_text(processed_img)
            
            print("\n" + "-"*60)
            print("📄 EXTRACTED TEXT:")
            print("-"*60)
            print(text)
            print("-"*60)
            
            # Step 3: Find potential drug names
            potential_drugs = self.extract_potential_drug_names(text)
            
            print(f"\n   🎯 Potential drugs: {potential_drugs}")
            
            # Step 4: Verify each drug
            verified_drugs = []
            
            print("\n   🔍 Verifying drugs with RxNorm database...")
            for drug in potential_drugs:
                result = self.fuzzy_match_drug(drug)
                if result['found']:
                    verified_drugs.append(result)
                    print(f"      ✅ {drug} → {result['name']} (RxCUI: {result['rxcui']})")
                else:
                    print(f"      ❌ {drug} → Not found in database")
            
            # Calculate confidence
            if len(potential_drugs) > 0:
                confidence = len(verified_drugs) / len(potential_drugs)
            else:
                confidence = 0
            
            # Results
            result = {
                'image': image_path,
                'raw_text': text,
                'potential_drugs': potential_drugs,
                'verified_drugs': verified_drugs,
                'confidence': confidence
            }
            
            print("\n" + "="*60)
            print("📊 SCAN RESULTS")
            print("="*60)
            print(f"Potential drugs found: {len(potential_drugs)}")
            print(f"Verified drugs: {len(verified_drugs)}")
            print(f"Confidence: {confidence*100:.1f}%")
            print("="*60)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return {
                'error': str(e),
                'image': image_path
            }


# ============================================================================
# DEMO / TESTING
# ============================================================================

def test_with_sample_text():
    """
    Test OCR with simulated prescription text
    (Use this if you don't have a prescription image)
    """
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE PRESCRIPTION TEXT")
    print("="*60)
    
    sample_text = """
    PRESCRIPTION
    
    Patient: John Doe
    Date: 05/07/2026
    
    Rx: LISINOPRIL 10mg
    Take 1 tablet daily
    
    Rx: METFORMIN 500mg
    Take 2 times daily with meals
    
    Rx: ATORVASTATIN 20mg
    Take 1 tablet at bedtime
    
    Dr. Smith, MD
    """
    
    ocr = PrescriptionOCR()
    
    print("\n📄 Sample Prescription:")
    print(sample_text)
    
    potential_drugs = ocr.extract_potential_drug_names(sample_text)
    print(f"\n🎯 Potential drugs found: {potential_drugs}")
    
    verified_drugs = []
    print("\n🔍 Verifying with RxNorm database...")
    for drug in potential_drugs:
        result = ocr.verify_drug_name(drug)
        if result['found']:
            verified_drugs.append(result)
            print(f"   ✅ {drug.upper()} verified! (RxCUI: {result['rxcui']})")
        else:
            print(f"   ❌ {drug} not found")
    
    print("\n" + "="*60)
    print(f"✅ Found {len(verified_drugs)} verified medications!")
    print("="*60)
    
    return verified_drugs


def create_test_image():
    """
    Create a simple test image with text
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = """LISINOPRIL 10mg
Take 1 tablet daily

METFORMIN 500mg
Take twice daily"""
    
    # Draw text
    draw.text((20, 20), text, fill='black')
    
    # Save
    img.save('test_prescription.png')
    print("✅ Created test image: test_prescription.png")
    
    return 'test_prescription.png'


if __name__ == "__main__":
    import sys
    
    print("\n💊 MedCheck OCR - Prescription Scanner")
    print("="*60)
    
    # Check if Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is installed!")
    except:
        print("\n❌ Tesseract not installed!")
        print("\nInstall instructions:")
        print("  Mac: brew install tesseract")
        print("  Ubuntu: sudo apt-get install tesseract-ocr")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nFor now, testing with text only...")
        test_with_sample_text()
        sys.exit(0)
    
    print("\n" + "="*60)
    print("TESTING OPTIONS:")
    print("="*60)
    print("1. Test with sample text (no image needed)")
    print("2. Create test image and scan it")
    print("3. Scan your own image (provide path)")
    print("="*60)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_with_sample_text()
    
    elif choice == "2":
        print("\n📸 Creating test image...")
        test_img = create_test_image()
        
        print("\n🔍 Scanning test image...")
        ocr = PrescriptionOCR()
        result = ocr.scan_prescription(test_img)
        
        if 'verified_drugs' in result and len(result['verified_drugs']) > 0:
            print("\n✅ SUCCESS! Drug names detected:")
            for drug in result['verified_drugs']:
                print(f"   💊 {drug['name'].upper()}")
    
    elif choice == "3":
        img_path = input("\nEnter image path: ").strip()
        
        ocr = PrescriptionOCR()
        result = ocr.scan_prescription(img_path)
        
        if 'verified_drugs' in result and len(result['verified_drugs']) > 0:
            print("\n✅ SUCCESS! Drug names detected:")
            for drug in result['verified_drugs']:
                print(f"   💊 {drug['name'].upper()}")
    
    else:
        print("Invalid choice. Running text test...")
        test_with_sample_text()
    
    print("\n" + "="*60)
    print("💡 NEXT STEPS:")
    print("   1. Try scanning real prescription images")
    print("   2. Improve accuracy by training custom models")
    print("   3. Add interaction checking")
    print("="*60)
