"""
Enhanced Image Preprocessing for OCR
Handles real-world photos (rotated, blurry, low contrast)

Improves OCR accuracy on phone photos!
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract

class EnhancedImagePreprocessor:
    """
    Advanced preprocessing for real-world prescription photos
    """
    
    def __init__(self):
        print("🎨 Enhanced Image Preprocessor Ready")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline
        """
        print(f"\n📸 Preprocessing: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print("   1️⃣ Rotation correction...")
        img = self._correct_rotation(img)
        
        print("   2️⃣ Contrast enhancement...")
        img = self._enhance_contrast(img)
        
        print("   3️⃣ Noise reduction...")
        img = self._reduce_noise(img)
        
        print("   4️⃣ Sharpening...")
        img = self._sharpen_image(img)
        
        print("   ✅ Preprocessing complete!")
        
        return img
    
    def _correct_rotation(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct text rotation
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect text orientation using Tesseract
        try:
            osd = pytesseract.image_to_osd(gray)
            angle = int(osd.split('Rotate: ')[1].split('\n')[0])
            
            if angle != 0:
                print(f"      📐 Detected rotation: {angle}°")
                
                # Rotate image
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        except:
            print("      ℹ️  Auto-rotation skipped")
        
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img
    
    def _reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce noise while preserving edges
        """
        # Bilateral filter - reduces noise, keeps edges
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        return img
    
    def _sharpen_image(self, img: np.ndarray) -> np.ndarray:
        """
        Sharpen text for better OCR
        """
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def save_preprocessed(self, img: np.ndarray, output_path: str):
        """
        Save preprocessed image
        """
        cv2.imwrite(output_path, img)
        print(f"   💾 Saved: {output_path}")


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo enhanced preprocessing"""
    
    print("\n" + "="*70)
    print("🎨 Enhanced Image Preprocessing Demo")
    print("="*70)
    print()
    print("Improvements:")
    print("  ✅ Rotation correction")
    print("  ✅ Contrast enhancement")
    print("  ✅ Noise reduction")
    print("  ✅ Image sharpening")
    print()
    print("Result: Better OCR on real-world photos!")
    print("="*70)
    print()
    
    preprocessor = EnhancedImagePreprocessor()
    
    # Test image
    test_image = "data/test_images/test_prescription.png"
    
    try:
        # Preprocess
        processed_img = preprocessor.preprocess_image(test_image)
        
        # Save result
        preprocessor.save_preprocessed(
            processed_img, 
            "data/test_images/test_prescription_enhanced.png"
        )
        
        print("\n✅ Preprocessing successful!")
        print("   Compare original vs enhanced image")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    demo()
