"""
Run the production API locally for testing
"""
import os
import sys
from app_production import app, categorizer

if __name__ == '__main__':
    print("🚀 Starting FinSaathi Production API...")
    print("🔗 API will be available at: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000")
    print("🧪 Test endpoint: http://localhost:5000/api/test")
    print("💡 Health check: http://localhost:5000/health")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Load models
    print("\n🔄 Loading trained models...")
    categorizer.load_models()
    
    if categorizer.is_loaded:
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Running with fallback models")
    
    print("\n🎉 Starting server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )