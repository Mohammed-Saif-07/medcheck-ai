"""
MedCheck - OCR Testing & Validation Framework

Features:
- Batch image testing
- Ground truth comparison
- Accuracy metrics calculation
- Error analysis & visualization
- Improvement recommendations

Usage:
1. Prepare test images in test_images/ folder
2. Create ground_truth.json with correct answers
3. Run testing
4. Get detailed accuracy report
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from pathlib import Path

# Import our advanced OCR
import sys
sys.path.insert(0, os.path.dirname(__file__))

class OCRTester:
    """
    Comprehensive testing framework for OCR system
    """
    
    def __init__(self, test_images_dir: str = "test_images"):
        self.test_images_dir = test_images_dir
        self.results = []
        self.ground_truth = {}
        
        # Create directories
        Path(test_images_dir).mkdir(exist_ok=True)
        Path("test_results").mkdir(exist_ok=True)
        
        print("🧪 MedCheck Testing Framework Initialized")
        print(f"📁 Test images directory: {test_images_dir}/")
        print(f"📊 Results directory: test_results/\n")
    
    def setup_ground_truth(self):
        """
        Interactive setup for ground truth labels
        """
        print("="*70)
        print("📝 GROUND TRUTH SETUP")
        print("="*70)
        print("\nThis creates the 'correct answers' for testing.\n")
        
        # Check for existing ground truth
        gt_file = "ground_truth.json"
        if os.path.exists(gt_file):
            print(f"✅ Found existing ground truth: {gt_file}")
            with open(gt_file, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"   Loaded {len(self.ground_truth)} test cases\n")
            
            update = input("Update ground truth? (y/n): ").lower()
            if update != 'y':
                return
        
        # Get list of test images
        image_files = [f for f in os.listdir(self.test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"\n❌ No images found in {self.test_images_dir}/")
            print("\n💡 Add prescription images to test_images/ folder")
            print("   Example images you can use:")
            print("   - test_prescription.png (already created)")
            print("   - Photos of real prescription bottles")
            print("   - Downloaded sample prescriptions\n")
            return
        
        print(f"\n📸 Found {len(image_files)} test images\n")
        
        # Collect ground truth for each image
        for img_file in image_files:
            if img_file in self.ground_truth:
                print(f"\n✅ {img_file}: Already labeled")
                print(f"   Drugs: {', '.join(self.ground_truth[img_file])}")
                continue
            
            print(f"\n📸 Image: {img_file}")
            print("   What drug names are in this image?")
            print("   Enter drug names separated by commas (or 'skip'):")
            
            drugs_input = input("   Drugs: ").strip()
            
            if drugs_input.lower() == 'skip':
                continue
            
            # Parse and clean drug names
            drugs = [d.strip().lower() for d in drugs_input.split(',') if d.strip()]
            
            if drugs:
                self.ground_truth[img_file] = drugs
                print(f"   ✅ Saved: {', '.join(drugs)}")
        
        # Save ground truth
        with open(gt_file, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        
        print(f"\n✅ Ground truth saved to {gt_file}")
        print(f"   Total test cases: {len(self.ground_truth)}\n")
    
    def run_batch_test(self, ocr_system):
        """
        Run OCR on all test images and compare with ground truth
        """
        print("="*70)
        print("🔬 RUNNING BATCH OCR TEST")
        print("="*70)
        
        if not self.ground_truth:
            print("\n❌ No ground truth data!")
            print("   Run setup_ground_truth() first\n")
            return
        
        total_images = len(self.ground_truth)
        print(f"\n📊 Testing {total_images} images...\n")
        
        for i, (img_file, expected_drugs) in enumerate(self.ground_truth.items(), 1):
            img_path = os.path.join(self.test_images_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"❌ Image not found: {img_path}")
                continue
            
            print(f"[{i}/{total_images}] Processing: {img_file}")
            
            # Run OCR
            start_time = time.time()
            try:
                result = ocr_system.process_image(img_path)
                processing_time = time.time() - start_time
                
                # Extract detected drugs
                detected_drugs = [
                    d.get('matched', d['name']).lower() 
                    for d in result.get('drugs_verified', [])
                ]
                
                # Calculate metrics for this image
                tp = len(set(detected_drugs) & set(expected_drugs))  # True Positives
                fp = len(set(detected_drugs) - set(expected_drugs))  # False Positives
                fn = len(set(expected_drugs) - set(detected_drugs))  # False Negatives
                
                # Store result
                test_result = {
                    'image': img_file,
                    'expected': expected_drugs,
                    'detected': detected_drugs,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'ocr_confidence': result.get('ocr_confidence', 0),
                    'processing_time': processing_time,
                    'correct': tp == len(expected_drugs) and fp == 0
                }
                
                self.results.append(test_result)
                
                # Print result
                if test_result['correct']:
                    print(f"   ✅ CORRECT: {', '.join(detected_drugs)}")
                else:
                    print(f"   ❌ INCORRECT:")
                    print(f"      Expected: {', '.join(expected_drugs)}")
                    print(f"      Detected: {', '.join(detected_drugs) if detected_drugs else 'None'}")
                    if fn > 0:
                        missed = set(expected_drugs) - set(detected_drugs)
                        print(f"      Missed: {', '.join(missed)}")
                    if fp > 0:
                        extra = set(detected_drugs) - set(expected_drugs)
                        print(f"      Extra: {', '.join(extra)}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.results.append({
                    'image': img_file,
                    'expected': expected_drugs,
                    'detected': [],
                    'error': str(e)
                })
            
            print()
        
        print("✅ Testing complete!\n")
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive accuracy metrics
        """
        if not self.results:
            return {}
        
        # Overall metrics
        total_tp = sum(r['true_positives'] for r in self.results)
        total_fp = sum(r['false_positives'] for r in self.results)
        total_fn = sum(r['false_negatives'] for r in self.results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Image-level accuracy
        correct_images = sum(1 for r in self.results if r.get('correct', False))
        image_accuracy = correct_images / len(self.results)
        
        # Average metrics
        avg_ocr_confidence = np.mean([r.get('ocr_confidence', 0) for r in self.results])
        avg_processing_time = np.mean([r.get('processing_time', 0) for r in self.results])
        
        metrics = {
            'total_images': len(self.results),
            'correct_images': correct_images,
            'image_accuracy': image_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'avg_ocr_confidence': avg_ocr_confidence,
            'avg_processing_time': avg_processing_time
        }
        
        return metrics
    
    def analyze_errors(self) -> Dict:
        """
        Detailed error analysis
        """
        error_analysis = {
            'common_misses': defaultdict(int),
            'common_false_detections': defaultdict(int),
            'low_confidence_images': [],
            'slow_images': []
        }
        
        for result in self.results:
            # Track missed drugs
            missed = set(result['expected']) - set(result['detected'])
            for drug in missed:
                error_analysis['common_misses'][drug] += 1
            
            # Track false detections
            extra = set(result['detected']) - set(result['expected'])
            for drug in extra:
                error_analysis['common_false_detections'][drug] += 1
            
            # Low confidence images
            if result.get('ocr_confidence', 1) < 0.7:
                error_analysis['low_confidence_images'].append({
                    'image': result['image'],
                    'confidence': result.get('ocr_confidence', 0)
                })
            
            # Slow processing
            if result.get('processing_time', 0) > 5:
                error_analysis['slow_images'].append({
                    'image': result['image'],
                    'time': result.get('processing_time', 0)
                })
        
        return error_analysis
    
    def generate_report(self):
        """
        Generate comprehensive testing report
        """
        print("="*70)
        print("📊 OCR TESTING REPORT")
        print("="*70)
        
        metrics = self.calculate_metrics()
        error_analysis = self.analyze_errors()
        
        # Overall Performance
        print("\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total Images Tested: {metrics['total_images']}")
        print(f"   Perfectly Correct: {metrics['correct_images']} ({metrics['image_accuracy']*100:.1f}%)")
        print()
        
        # Drug-Level Metrics
        print("💊 DRUG-LEVEL METRICS:")
        print(f"   Precision: {metrics['precision']*100:.1f}%")
        print(f"   Recall: {metrics['recall']*100:.1f}%")
        print(f"   F1-Score: {metrics['f1_score']*100:.1f}%")
        print()
        print(f"   True Positives: {metrics['true_positives']} (correctly detected)")
        print(f"   False Positives: {metrics['false_positives']} (incorrectly detected)")
        print(f"   False Negatives: {metrics['false_negatives']} (missed)")
        print()
        
        # Performance
        print("⚡ PERFORMANCE:")
        print(f"   Avg OCR Confidence: {metrics['avg_ocr_confidence']*100:.1f}%")
        print(f"   Avg Processing Time: {metrics['avg_processing_time']:.2f}s")
        print()
        
        # Error Analysis
        if error_analysis['common_misses']:
            print("❌ COMMONLY MISSED DRUGS:")
            for drug, count in sorted(error_analysis['common_misses'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {drug.upper()}: missed {count} time(s)")
            print()
        
        if error_analysis['common_false_detections']:
            print("⚠️  COMMON FALSE DETECTIONS:")
            for drug, count in sorted(error_analysis['common_false_detections'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {drug}: detected incorrectly {count} time(s)")
            print()
        
        # Recommendations
        print("="*70)
        print("💡 RECOMMENDATIONS:")
        print("="*70)
        
        if metrics['f1_score'] >= 0.85:
            print("\n✅ EXCELLENT! OCR system is production-ready!")
            print("   Next steps:")
            print("   1. Build ML models (interaction severity)")
            print("   2. Add RAG validation system")
            print("   3. Integrate LLM for explanations")
        elif metrics['f1_score'] >= 0.70:
            print("\n🟡 GOOD! OCR system is working but needs improvement.")
            print("   Recommended improvements:")
            print("   1. Add data augmentation (rotation, brightness)")
            print("   2. Fine-tune preprocessing parameters")
            print("   3. Test more OCR models (PaddleOCR)")
        else:
            print("\n🔴 NEEDS IMPROVEMENT! OCR accuracy is too low.")
            print("   Critical improvements needed:")
            print("   1. Collect more diverse training images")
            print("   2. Implement data augmentation pipeline")
            print("   3. Consider custom CNN for layout detection")
            print("   4. Manual review and correction loop")
        
        print("\n" + "="*70)
        
        # Save detailed results
        results_file = "test_results/detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'error_analysis': {
                    'common_misses': dict(error_analysis['common_misses']),
                    'common_false_detections': dict(error_analysis['common_false_detections']),
                    'low_confidence_images': error_analysis['low_confidence_images'],
                    'slow_images': error_analysis['slow_images']
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}\n")
        
        return metrics
    
    def visualize_results(self):
        """
        Create visualization charts
        """
        if not self.results:
            print("No results to visualize!")
            return
        
        metrics = self.calculate_metrics()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MedCheck OCR Testing Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Bar Chart
        ax1 = axes[0, 0]
        categories = ['Precision', 'Recall', 'F1-Score', 'Image\nAccuracy']
        values = [
            metrics['precision']*100,
            metrics['recall']*100,
            metrics['f1_score']*100,
            metrics['image_accuracy']*100
        ]
        colors = ['#2ecc71' if v >= 85 else '#f39c12' if v >= 70 else '#e74c3c' for v in values]
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Accuracy Metrics')
        ax1.set_ylim(0, 100)
        ax1.axhline(y=85, color='g', linestyle='--', alpha=0.3, label='Excellent (85%+)')
        ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.3, label='Good (70%+)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Confusion Matrix Style
        ax2 = axes[0, 1]
        confusion_data = [
            ['True\nPositives', metrics['true_positives']],
            ['False\nPositives', metrics['false_positives']],
            ['False\nNegatives', metrics['false_negatives']]
        ]
        
        y_pos = np.arange(len(confusion_data))
        values = [item[1] for item in confusion_data]
        labels = [item[0] for item in confusion_data]
        colors_conf = ['#2ecc71', '#e74c3c', '#e74c3c']
        
        bars = ax2.barh(y_pos, values, color=colors_conf, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Count')
        ax2.set_title('Detection Breakdown')
        
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {value}', ha='left', va='center', fontweight='bold')
        
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Processing Time Distribution
        ax3 = axes[1, 0]
        processing_times = [r.get('processing_time', 0) for r in self.results]
        ax3.hist(processing_times, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Processing Time (seconds)')
        ax3.set_ylabel('Number of Images')
        ax3.set_title(f'Processing Time Distribution\n(Avg: {metrics["avg_processing_time"]:.2f}s)')
        ax3.axvline(x=metrics['avg_processing_time'], color='r', linestyle='--', 
                   linewidth=2, label=f'Average: {metrics["avg_processing_time"]:.2f}s')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. OCR Confidence Distribution
        ax4 = axes[1, 1]
        confidences = [r.get('ocr_confidence', 0)*100 for r in self.results]
        ax4.hist(confidences, bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('OCR Confidence (%)')
        ax4.set_ylabel('Number of Images')
        ax4.set_title(f'OCR Confidence Distribution\n(Avg: {metrics["avg_ocr_confidence"]*100:.1f}%)')
        ax4.axvline(x=metrics['avg_ocr_confidence']*100, color='r', linestyle='--',
                   linewidth=2, label=f'Average: {metrics["avg_ocr_confidence"]*100:.1f}%')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        chart_file = 'test_results/testing_results.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {chart_file}")
        
        plt.show()


# ============================================================================
# MAIN TESTING WORKFLOW
# ============================================================================

def main():
    """Main testing workflow"""
    
    print("\n💊 MedCheck - OCR Testing Framework")
    print("="*70)
    print("\nThis will test your OCR system on real prescription images.\n")
    
    # Initialize tester
    tester = OCRTester()
    
    # Setup ground truth
    print("Step 1: Setup ground truth (correct answers)")
    tester.setup_ground_truth()
    
    if not tester.ground_truth:
        print("\n❌ No test data available!")
        print("\n💡 TO GET STARTED:")
        print("   1. Add prescription images to 'test_images/' folder")
        print("   2. Run this script again")
        print("   3. Label each image with correct drug names")
        print("\n   You can use:")
        print("   - test_prescription.png (already created)")
        print("   - Your own prescription bottle photos")
        print("   - Downloaded sample prescriptions from internet\n")
        return
    
    # Import OCR system
    print("\nStep 2: Loading OCR system...")
    try:
        from advanced_ocr import AdvancedOCR
        ocr_system = AdvancedOCR()
    except Exception as e:
        print(f"❌ Error loading OCR system: {e}")
        print("   Make sure advanced_ocr.py is in the same directory")
        return
    
    # Run tests
    print("\nStep 3: Running batch OCR tests...")
    tester.run_batch_test(ocr_system)
    
    # Generate report
    print("\nStep 4: Generating report...")
    metrics = tester.generate_report()
    
    # Create visualizations
    try:
        print("\nStep 5: Creating visualizations...")
        tester.visualize_results()
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
        print("   Install matplotlib: pip install matplotlib seaborn")
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 TESTING COMPLETE!")
    print("="*70)
    print(f"\n📊 F1-Score: {metrics['f1_score']*100:.1f}%")
    
    if metrics['f1_score'] >= 0.85:
        print("   ✅ EXCELLENT - Ready for ML models!")
    elif metrics['f1_score'] >= 0.70:
        print("   🟡 GOOD - Consider improvements before ML")
    else:
        print("   🔴 NEEDS WORK - Improve OCR first")
    
    print("\n💡 Next: Check test_results/ folder for detailed analysis\n")


if __name__ == "__main__":
    main()
