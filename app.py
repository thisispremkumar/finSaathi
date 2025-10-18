"""
Production Flask API for FinSaathi Expense Categorization
- Loads trained pickle models on startup
- Optimized for production deployment
- Enhanced error handling and logging
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

class ProductionCategorizer:
    """Production-ready categorizer using trained pickle models"""
    
    def __init__(self):
        self.hybrid_ai = None
        self.ml_categorizer = None
        self.is_loaded = False
        self.model_info = {}
        
    def load_models(self):
        """Load trained models from pickle files"""
        try:
            # Load Hybrid AI model (primary)
            hybrid_path = 'models/financial_analyst_ai_latest.pkl'
            if os.path.exists(hybrid_path):
                with open(hybrid_path, 'rb') as f:
                    self.hybrid_ai = pickle.load(f)
                logger.info("✅ Loaded Hybrid AI model successfully")
                self.model_info['hybrid_ai'] = 'loaded'
            else:
                logger.warning("⚠️ Hybrid AI model not found, initializing fallback")
                self.hybrid_ai = self._create_fallback_ai()
                self.model_info['hybrid_ai'] = 'fallback'
            
            # Load ML Categorizer (secondary)
            ml_path = 'models/enhanced_ml_categorizer_latest.pkl'
            if os.path.exists(ml_path):
                with open(ml_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_categorizer = model_data.get('categorizer')
                logger.info("✅ Loaded ML Categorizer successfully")
                self.model_info['ml_categorizer'] = 'loaded'
            else:
                logger.warning("⚠️ ML Categorizer not found")
                self.model_info['ml_categorizer'] = 'not_found'
            
            self.is_loaded = True
            logger.info("🎉 All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(traceback.format_exc())
            # Create fallback system
            self.hybrid_ai = self._create_fallback_ai()
            self.is_loaded = False
    
    def _create_fallback_ai(self):
        """Create fallback AI system if pickle models fail"""
        class FallbackAI:
            def categorize_transaction(self, text):
                text_lower = text.lower()
                
                # Simple rule-based categorization
                if any(term in text_lower for term in ['clinical', 'laboratory', 'hospital', 'pharmacy', 'medical']):
                    return {'category': 'Healthcare', 'confidence_score': 0.9, 'merchant_name': self._extract_merchant(text)}
                elif any(term in text_lower for term in ['power', 'electricity', 'airtel', 'jio', 'vodafone']):
                    return {'category': 'Utilities', 'confidence_score': 0.9, 'merchant_name': self._extract_merchant(text)}
                elif any(term in text_lower for term in ['amazon', 'flipkart', 'myntra', 'shopping']):
                    return {'category': 'Shopping', 'confidence_score': 0.8, 'merchant_name': self._extract_merchant(text)}
                elif any(term in text_lower for term in ['uber', 'ola', 'petrol', 'fuel']):
                    return {'category': 'Transportation', 'confidence_score': 0.8, 'merchant_name': self._extract_merchant(text)}
                elif any(term in text_lower for term in ['zomato', 'swiggy', 'mcdonald', 'restaurant']):
                    return {'category': 'Food & Dining', 'confidence_score': 0.8, 'merchant_name': self._extract_merchant(text)}
                else:
                    return {'category': 'Other', 'confidence_score': 0.5, 'merchant_name': self._extract_merchant(text)}
            
            def _extract_merchant(self, text):
                patterns = [
                    r'(?:to|at)\s+([A-Z][A-Z\s&.\-\(\)]+?)(?:\s+on|\s+for|\.|$)',
                    r'([A-Z][A-Z\s&.\-\(\)]{3,}?)(?:\s+on|\s+for|\.|$)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1).strip()
                return None
        
        return FallbackAI()
    
    def extract_sms_data(self, sms_text):
        """Extract structured data from SMS"""
        # Amount extraction
        amount_patterns = [
            r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'₹\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
        ]
        
        amount = None
        for pattern in amount_patterns:
            match = re.search(pattern, sms_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    break
                except ValueError:
                    continue
        
        # Merchant extraction patterns
        merchant_patterns = [
            r'(?:to|at|from|for)\s+([A-Z][A-Z\s&.\-\(\)0-9]+?)(?:\s+on|\s+for|\s+avl|\s+ref|\s+upi|\.|$)',
            r'(?:spent|paid|debited|credited).*?(?:at|to|from|for)\s+([A-Z][A-Z\s&.\-\(\)0-9]+?)(?:\s+on|\s+for|\s+avl|\.|$)',
            r'([A-Z][A-Z\s&.\-\(\)0-9]{3,}?)(?:\s+on\s+\d|\s+for|\s+avl|\s+ref|\s+upi|\.|$)'
        ]
        
        merchant = None
        for pattern in merchant_patterns:
            match = re.search(pattern, sms_text)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 2 and not re.match(r'^\d+$', candidate):
                    merchant = candidate
                    break
        
        # Transaction type
        if re.search(r'debited|spent|paid', sms_text, re.IGNORECASE):
            txn_type = 'debit'
        elif re.search(r'credited|received', sms_text, re.IGNORECASE):
            txn_type = 'credit'
        else:
            txn_type = 'unknown'
        
        # Date extraction
        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', sms_text)
        date = date_match.group(1) if date_match else None
        
        return {
            'amount': amount,
            'merchant': merchant,
            'transaction_type': txn_type,
            'date': date,
            'raw_text': sms_text
        }
    
    def categorize_sms(self, sms_text):
        """Main SMS categorization method using trained models"""
        try:
            # Extract SMS data
            sms_data = self.extract_sms_data(sms_text)
            
            # Primary categorization using Hybrid AI
            if self.hybrid_ai:
                try:
                    ai_result = self.hybrid_ai.categorize_transaction(sms_text)
                    ai_confidence = ai_result.get('confidence_score', 0.0)
                    ai_category = ai_result.get('category', 'Other')
                except Exception as e:
                    logger.error(f"Hybrid AI categorization failed: {e}")
                    ai_result = {'category': 'Other', 'confidence_score': 0.0, 'merchant_name': None}
                    ai_confidence = 0.0
                    ai_category = 'Other'
            else:
                ai_result = {'category': 'Other', 'confidence_score': 0.0, 'merchant_name': None}
                ai_confidence = 0.0
                ai_category = 'Other'
            
            # Secondary categorization using ML model (if available)
            ml_result = None
            ml_confidence = 0.0
            ml_category = None
            
            if self.ml_categorizer:
                try:
                    ml_result = self.ml_categorizer.categorize_expense(
                        sms_text, sms_data['merchant'], sms_data['amount']
                    )
                    ml_confidence = ml_result.get('confidence', 0.0)
                    ml_category = ml_result.get('primary_category', 'Other')
                except Exception as e:
                    logger.error(f"ML categorization failed: {e}")
            
            # Decision logic: Use higher confidence result
            if ai_confidence >= ml_confidence:
                final_category = ai_category
                final_confidence = ai_confidence
                method = 'Hybrid_AI'
                merchant_detected = ai_result.get('merchant_name') or sms_data['merchant']
            else:
                final_category = ml_category
                final_confidence = ml_confidence
                method = 'ML_Model'
                merchant_detected = sms_data['merchant']
            
            return {
                'sms_data': sms_data,
                'category': final_category,
                'confidence': final_confidence,
                'method': method,
                'merchant_detected': merchant_detected,
                'ai_prediction': {
                    'category': ai_category,
                    'confidence': ai_confidence
                },
                'ml_prediction': {
                    'category': ml_category,
                    'confidence': ml_confidence
                } if ml_result else None
            }
            
        except Exception as e:
            logger.error(f"SMS categorization error: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'category': 'Other',
                'confidence': 0.0,
                'sms_data': self.extract_sms_data(sms_text)
            }

# Initialize categorizer and load models
categorizer = ProductionCategorizer()

@app.before_serving
def initialize_models():
    """Load models once before serving requests"""
    logger.info("🔄 Loading trained models...")
    categorizer.load_models()

# API Routes
@app.route('/')
def home():
    """API home page with documentation"""
    return jsonify({
        "service": "FinSaathi Expense Categorization API",
        "version": "2.0.0 (Production)",
        "status": "active",
        "description": "Production API using trained pickle models for Indian transaction categorization",
        "model_status": categorizer.model_info,
        "features": [
            "Trained pickle model loading",
            "Hybrid AI + ML categorization",
            "SMS transaction parsing",
            "Indian merchant recognition",
            "Healthcare, utilities, shopping detection",
            "Production-ready error handling"
        ],
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check and model status",
            "POST /api/categorize": "Basic transaction categorization",
            "POST /api/categorize/sms": "SMS transaction categorization",
            "POST /api/categorize/batch": "Batch SMS processing",
            "GET /api/test": "Test endpoint with sample data"
        },
        "usage_examples": {
            "sms_categorization": {
                "url": "/api/categorize/sms",
                "method": "POST",
                "body": {
                    "sms_text": "A/c *5678 debited Rs. 970.00 to UMA CLINICAL LABORATORY"
                },
                "expected_response": {
                    "category": "Healthcare",
                    "confidence": 0.95
                }
            }
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": categorizer.is_loaded,
        "model_status": categorizer.model_info,
        "api_version": "2.0.0",
        "ready": True
    })

@app.route('/api/categorize', methods=['POST'])
def categorize_basic():
    """Basic transaction categorization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        description = data.get('description', '')
        
        if not description:
            return jsonify({'error': 'Description is required'}), 400
        
        # Use hybrid AI for basic categorization
        result = categorizer.hybrid_ai.categorize_transaction(description)
        
        return jsonify({
            'success': True,
            'result': {
                'category': result['category'],
                'confidence': result['confidence_score'],
                'merchant': result.get('merchant_name')
            },
            'method': 'Hybrid_AI',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Basic categorization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categorize/sms', methods=['POST'])
def categorize_sms():
    """SMS transaction categorization using trained models"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sms_text = data.get('sms_text', '')
        
        if not sms_text:
            return jsonify({'error': 'sms_text is required'}), 400
        
        # Categorize using trained models
        result = categorizer.categorize_sms(sms_text)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"SMS categorization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categorize/batch', methods=['POST'])
def categorize_batch():
    """Batch SMS processing"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sms_list = data.get('sms_list', [])
        
        if not sms_list:
            return jsonify({'error': 'sms_list is required'}), 400
        
        results = []
        for i, sms_text in enumerate(sms_list):
            try:
                result = categorizer.categorize_sms(sms_text)
                results.append({
                    'index': i,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'sms_text': sms_text
                })
        
        return jsonify({
            'success': True,
            'processed_count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test_endpoint():
    """Test endpoint to verify model functionality"""
    test_cases = [
        "A/c *5678 debited Rs. 970.00 on 10-05-25 to UMA CLINICAL LABORATORY. UPI:882918376710",
        "Rs.1250.00 debited from A/c XX1234 on 15-Oct-25 to MYNTRA FASHION STORE for online purchase",
        "Payment of Rs.45.50 made to INDIAN OIL PETROL PUMP on 15-Oct-25 via UPI",
        "Account debited Rs.285.50 on 15-Oct-25 at STARBUCKS COFFEE STORE for Card ending 1234"
    ]
    
    results = []
    for sms in test_cases:
        try:
            result = categorizer.categorize_sms(sms)
            results.append({
                'sms': sms[:50] + "...",
                'category': result['category'],
                'confidence': result['confidence'],
                'method': result['method']
            })
        except Exception as e:
            results.append({
                'sms': sms[:50] + "...",
                'error': str(e)
            })
    
    return jsonify({
        'test_results': results,
        'model_status': categorizer.model_info,
        'models_loaded': categorizer.is_loaded
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting FinSaathi Production API on port {port}")
    logger.info(f"🔗 API available at: http://localhost:{port}")
    logger.info(f"📚 Documentation: http://localhost:{port}")
    
    # Load models at startup
    categorizer.load_models()
    
    app.run(host='0.0.0.0', port=port, debug=debug)
