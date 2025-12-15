"""
Verification script for HealthInsight AI
Tests all components and confirms installation
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import tensorflow
        import sklearn
        import nltk
        import Bio
        import matplotlib
        import seaborn
        import pandas
        import numpy
        print("✅ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    sys.path.insert(0, 'src')
    
    try:
        from data_generator import MedicalDataGenerator
        from dl_model import DiseasePredictionModel
        from nlp_analyzer import MedicalNLPAnalyzer
        from bio_analyzer import BiologicalDataAnalyzer
        from visualizer import HealthcareVisualizer
        print("✅ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Module error: {e}")
        return False

def test_functionality():
    """Test basic functionality of each component"""
    print("\nTesting component functionality...")
    
    from data_generator import MedicalDataGenerator
    from nlp_analyzer import MedicalNLPAnalyzer
    from bio_analyzer import BiologicalDataAnalyzer
    
    # Test data generation
    gen = MedicalDataGenerator(n_samples=10)
    data = gen.generate_patient_data()
    assert len(data) == 10, "Data generation failed"
    print("  ✅ Data generator working")
    
    # Test NLP
    nlp = MedicalNLPAnalyzer()
    symptoms = nlp.extract_symptoms("Patient has chest pain")
    assert len(symptoms) > 0, "NLP extraction failed"
    print("  ✅ NLP analyzer working")
    
    # Test Biopython
    bio = BiologicalDataAnalyzer()
    seqs = bio.generate_sample_sequences(3)
    assert len(seqs) == 3, "Sequence generation failed"
    print("  ✅ Biological analyzer working")
    
    return True

def check_files():
    """Check if expected files exist"""
    print("\nChecking project structure...")
    
    required_files = [
        'main.py',
        'examples.py',
        'requirements.txt',
        'README.md',
        'TEST_RESULTS.md',
        'SUMMARY.md',
        'src/data_generator.py',
        'src/dl_model.py',
        'src/nlp_analyzer.py',
        'src/bio_analyzer.py',
        'src/visualizer.py'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification tests"""
    print("="*70)
    print(" "*15 + "HEALTHINSIGHT AI VERIFICATION")
    print("="*70)
    
    tests = [
        ("Dependencies", test_imports),
        ("Custom Modules", test_modules),
        ("Functionality", test_functionality),
        ("Project Structure", check_files)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Installation verified successfully!")
        print("\nYou can now:")
        print("  1. Run full pipeline: python main.py")
        print("  2. Try examples: python examples.py")
        print("  3. Read documentation: README.md, TEST_RESULTS.md")
    else:
        print("⚠️  Some tests failed - please check the errors above")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
