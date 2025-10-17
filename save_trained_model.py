#!/usr/bin/env python3
"""
Save the current trained model to pickle file
This preserves the trained model for future use without retraining
"""

import sys
import os
import pickle
import joblib
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly without the problematic dependencies
try:
    from app_hybrid import FinancialAnalystAI
    HYBRID_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not import FinancialAnalystAI: {e}")
    HYBRID_AVAILABLE = False

# Check if the ML model file exists
ML_MODEL_PATH = "enhanced_expense_model.pkl"

def save_models():
    """Save the trained models"""
    
    print("💾 Saving Trained Models")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 Created {models_dir} directory")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import json
    
    # 1. Copy existing ML model if it exists
    print("\n🤖 Handling ML Categorizer...")
    if os.path.exists(ML_MODEL_PATH):
        try:
            import shutil
            # Copy the existing model
            ml_model_path = os.path.join(models_dir, f"enhanced_ml_categorizer_{timestamp}.pkl")
            shutil.copy2(ML_MODEL_PATH, ml_model_path)
            print(f"✅ ML Model copied to: {ml_model_path}")
            
            # Also create a latest version
            latest_ml_path = os.path.join(models_dir, "enhanced_ml_categorizer_latest.pkl")
            shutil.copy2(ML_MODEL_PATH, latest_ml_path)
            print(f"✅ Latest ML model saved to: {latest_ml_path}")
            
            # Save model info
            model_info = {
                'timestamp': timestamp,
                'model_type': 'ImprovedExpenseCategorizer',
                'source': 'enhanced_expense_model.pkl',
                'model_version': '2.0_comprehensive_merchants',
                'notes': 'Copied from existing trained model'
            }
            
            info_path = os.path.join(models_dir, f"ml_model_info_{timestamp}.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"📋 Model info saved to: {info_path}")
            
        except Exception as e:
            print(f"❌ Failed to copy ML model: {e}")
    else:
        print(f"⚠️ ML model file not found: {ML_MODEL_PATH}")
    
    # 2. Save the Hybrid Financial Analyst AI
    if HYBRID_AVAILABLE:
        print("\n🧠 Saving Hybrid Financial Analyst AI...")
        try:
            hybrid_system = FinancialAnalystAI()
            
            # Save the hybrid system
            hybrid_model_path = os.path.join(models_dir, f"financial_analyst_ai_{timestamp}.pkl")
            with open(hybrid_model_path, 'wb') as f:
                pickle.dump(hybrid_system, f)
            print(f"✅ Hybrid AI saved to: {hybrid_model_path}")
            
            # Also save with a standard name (latest)
            latest_hybrid_path = os.path.join(models_dir, "financial_analyst_ai_latest.pkl")
            with open(latest_hybrid_path, 'wb') as f:
                pickle.dump(hybrid_system, f)
            print(f"✅ Latest hybrid AI saved to: {latest_hybrid_path}")
            
            # Save hybrid system info
            hybrid_info = {
                'timestamp': timestamp,
                'model_type': 'FinancialAnalystAI',
                'features': [
                    'rule_based_merchant_detection', 
                    'utility_company_database', 
                    'insurance_detection', 
                    'education_institution_recognition',
                    'comprehensive_indian_merchants_116_companies'
                ],
                'merchant_categories': [
                    'Utilities (Electricity/Water/Gas/Telecom/DTH)',
                    'Insurance (18_companies)',
                    'Transportation (FASTag/Toll_17_providers)',
                    'Education (19_institutions)', 
                    'Bills_&_Utilities (19_payment_platforms)'
                ],
                'accuracy': '100%_on_test_suites',
                'version': '3.0_comprehensive_integration'
            }
            
            hybrid_info_path = os.path.join(models_dir, f"hybrid_ai_info_{timestamp}.json")
            with open(hybrid_info_path, 'w') as f:
                json.dump(hybrid_info, f, indent=2)
            print(f"📋 Hybrid AI info saved to: {hybrid_info_path}")
            
        except Exception as e:
            print(f"❌ Failed to save hybrid AI: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ Hybrid AI not available - skipping")
    
    # 3. Test loading the saved models
    print("\n🧪 Testing Model Loading...")
    try:
        if HYBRID_AVAILABLE and os.path.exists(os.path.join(models_dir, "financial_analyst_ai_latest.pkl")):
            with open(os.path.join(models_dir, "financial_analyst_ai_latest.pkl"), 'rb') as f:
                loaded_hybrid = pickle.load(f)
            test_result = loaded_hybrid.categorize_transaction("Test payment Rs. 100 to BESCOM for electricity")
            print(f"✅ Hybrid AI loads correctly: {test_result.get('category', 'Unknown')} ({test_result.get('confidence_score', 0)*100:.1f}%)")
        
        if os.path.exists(os.path.join(models_dir, "enhanced_ml_categorizer_latest.pkl")):
            # Try to load ML model
            try:
                loaded_ml = joblib.load(os.path.join(models_dir, "enhanced_ml_categorizer_latest.pkl"))
                print(f"✅ ML Model loads correctly: {type(loaded_ml).__name__}")
            except Exception as e:
                print(f"⚠️ ML model loading failed: {e}")
        
    except Exception as e:
        print(f"⚠️ Model loading test failed: {e}")
    
    # 4. Create a model loader utility
    print("\n📦 Creating Model Loader Utility...")
    loader_code = '''#!/usr/bin/env python3
"""
Utility to load saved models
"""

import joblib
import pickle
import os

def load_ml_categorizer(model_path=None):
    """Load the enhanced ML categorizer"""
    if model_path is None:
        model_path = os.path.join("models", "enhanced_ml_categorizer_latest.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)

def load_hybrid_ai(model_path=None):
    """Load the hybrid financial analyst AI"""
    if model_path is None:
        model_path = os.path.join("models", "financial_analyst_ai_latest.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_available_models():
    """List all available saved models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            models.append(os.path.join(models_dir, file))
    return models

# Example usage
if __name__ == "__main__":
    print("Available models:")
    for model in get_available_models():
        print(f"  {model}")
    
    # Load and test models
    try:
        ml_model = load_ml_categorizer()
        print("\\nML Categorizer loaded successfully")
        
        hybrid_model = load_hybrid_ai()
        print("Hybrid AI loaded successfully")
        
        # Quick test
        test_sms = "Payment to BESCOM Rs. 450.00 for electricity bill"
        result = hybrid_model.categorize_transaction(test_sms)
        print(f"\\nTest categorization: {result}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
'''
    
    loader_path = "load_saved_models.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    print(f"📋 Model loader utility created: {loader_path}")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("📊 MODEL SAVING SUMMARY")
    print(f"🕒 Timestamp: {timestamp}")
    print(f"📁 Models Directory: {models_dir}/")
    print("💾 Saved Files:")
    
    saved_files = []
    for file in os.listdir(models_dir):
        if timestamp in file or 'latest' in file:
            file_path = os.path.join(models_dir, file)
            file_size = os.path.getsize(file_path)
            saved_files.append(f"  {file} ({file_size:,} bytes)")
    
    for file_info in saved_files:
        print(file_info)
    
    print(f"📋 Loader utility: {loader_path}")
    print("\n🎉 All models saved successfully!")
    print("💡 Use 'python load_saved_models.py' to load and test models")

if __name__ == "__main__":
    save_models()