#!/usr/bin/env python3
"""
Utility to load and test saved pickle models
"""

import pickle
import os
from datetime import datetime

def load_hybrid_ai(model_path='models/financial_analyst_ai_latest.pkl'):
    """Load the hybrid AI model from pickle"""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded Hybrid AI model from {model_path}")
            return model
        else:
            print(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_ml_categorizer(model_path='models/enhanced_ml_categorizer_latest.pkl'):
    """Load the ML categorizer from pickle"""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"✅ Loaded ML Categorizer from {model_path}")
            return model_data.get('categorizer')
        else:
            print(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_models():
    """Test both models with sample data"""
    print("🧪 Testing saved models...")
    
    # Test SMS
    test_sms = "A/c *5678 debited Rs. 970.00 on 10-05-25 to UMA CLINICAL LABORATORY"
    
    # Test Hybrid AI
    hybrid_ai = load_hybrid_ai()
    if hybrid_ai:
        try:
            result = hybrid_ai.categorize_transaction(test_sms)
            print(f"🎯 Hybrid AI Result: {result['category']} ({result['confidence_score']:.2f})")
        except Exception as e:
            print(f"❌ Hybrid AI test failed: {e}")
    
    # Test ML Categorizer
    ml_model = load_ml_categorizer()
    if ml_model:
        try:
            result = ml_model.categorize_expense(test_sms, "UMA CLINICAL LABORATORY", 970.0)
            print(f"🎯 ML Model Result: {result['primary_category']} ({result['confidence']:.2f})")
        except Exception as e:
            print(f"❌ ML Model test failed: {e}")

if __name__ == '__main__':
    test_models()
