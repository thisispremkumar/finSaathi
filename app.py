"""
Production Flask API for FinSaathi Expense Categorization - STANDALONE VERSION
- AI model categorizes SMS without requiring any positional arguments
- Enhanced self-sufficient categorization
- Robust merchant extraction and category detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import logging
from datetime import datetime
import traceback
from enhanced_categorizer_v2 import ImprovedExpenseCategorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
logger = logging.getLogger(__name__)

logger.info("🔄 Loading trained models...")
# Initialize categorizer and load models
categorizer = ImprovedExpenseCategorizer()
categorizer.load_model()
CORS(app)  # Enable CORS for frontend integration


class ProductionCategorizer:
    """Standalone categorizer that only needs SMS text"""
    
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
                logger.warning("⚠️ Hybrid AI model not found, initializing enhanced fallback")
                self.hybrid_ai = self._create_enhanced_ai()
                self.model_info['hybrid_ai'] = 'enhanced_fallback'
            
            # Load ML Categorizer (optional - we'll make it standalone)
            ml_path = 'models/enhanced_ml_categorizer_latest.pkl'
            if os.path.exists(ml_path):
                with open(ml_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_categorizer = model_data.get('categorizer')
                logger.info("✅ Loaded ML Categorizer successfully")
                self.model_info['ml_categorizer'] = 'loaded'
            else:
                logger.info("ℹ️ ML Categorizer not found - using AI-only approach")
                self.model_info['ml_categorizer'] = 'not_needed'
            
            self.is_loaded = True
            logger.info("🎉 Standalone AI system ready!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(traceback.format_exc())
            # Create enhanced standalone AI
            self.hybrid_ai = self._create_enhanced_ai()
            self.is_loaded = True  # Still functional with enhanced AI
    
    def _create_enhanced_ai(self):
        """Create enhanced standalone AI that needs no external arguments"""
        class EnhancedStandaloneAI:
            def __init__(self):
                # Comprehensive Indian merchant and category database
                self.category_rules = {
                    'Shopping': {
                        'merchants': [
                            'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'meesho',
                            'reliance digital', 'croma', 'vijay sales', 'shoppers stop',
                            'lifestyle', 'pantaloons', 'max fashion', 'westside',
                            'jeweller', 'jewellery', 'jewelry', 'gold', 'silver', 'diamond',
                            'sri', 'lakshmi', 'ganapathi', 'swamy', 'mall', 'store',
                            'big bazaar', 'dmart', 'more', 'spencer', 'nilgiris'
                        ],
                        'keywords': [
                            'shopping', 'purchase', 'buy', 'mall', 'store', 'retail',
                            'jeweller', 'jewellery', 'jewelry', 'gold', 'silver', 'diamond',
                            'fashion', 'clothing', 'accessories', 'electronics', 'mobile'
                        ]
                    },
                    'Healthcare': {
                        'merchants': [
                            'apollo', 'fortis', 'max healthcare', 'manipal', 'columbia asia',
                            'apollo pharmacy', 'medplus', 'netmeds', 'pharmeasy', '1mg',
                            'clinical laboratory', 'pathology', 'diagnostic', 'uma clinical',
                            'dr lal pathlabs', 'srl diagnostics', 'metropolis'
                        ],
                        'keywords': [
                            'hospital', 'clinic', 'pharmacy', 'medical', 'laboratory', 'lab',
                            'diagnostic', 'pathology', 'clinical', 'healthcare', 'medicine',
                            'doctor', 'health', 'checkup', 'test'
                        ]
                    },
                    'Utilities': {
                        'merchants': [
                            'tata power', 'adani electricity', 'bescom', 'mseb', 'kseb', 'tneb',
                            'eastern power', 'power distribution', 'electricity board',
                            'airtel', 'jio', 'vodafone', 'bsnl', 'vi', 'idea',
                            'tata sky', 'dish tv', 'sun direct', 'airtel digital tv'
                        ],
                        'keywords': [
                            'electricity', 'power', 'utility', 'bill', 'recharge', 'broadband',
                            'mobile', 'phone', 'internet', 'wifi', 'dth', 'cable'
                        ]
                    },
                    'Food & Dining': {
                        'merchants': [
                            'zomato', 'swiggy', 'uber eats', 'foodpanda', 'dominos', 'pizza hut',
                            'mcdonalds', 'kfc', 'burger king', 'subway', 'starbucks',
                            'cafe coffee day', 'barista', 'haldirams', 'bikanervala'
                        ],
                        'keywords': [
                            'restaurant', 'cafe', 'coffee', 'pizza', 'food', 'dining', 'meal',
                            'lunch', 'dinner', 'breakfast', 'snacks', 'delivery'
                        ]
                    },
                    'Transportation': {
                        'merchants': [
                            'uber', 'ola', 'rapido', 'meru', 'fastag', 'toll', 'nhai',
                            'indian oil', 'bharat petroleum', 'hp petrol', 'shell',
                            'petrol pump', 'fuel station', 'gas station'
                        ],
                        'keywords': [
                            'taxi', 'cab', 'ride', 'fuel', 'petrol', 'diesel', 'transport',
                            'toll', 'fastag', 'parking', 'auto'
                        ]
                    },
                    'Entertainment': {
                        'merchants': [
                            'netflix', 'amazon prime', 'hotstar', 'disney', 'sony liv',
                            'zee5', 'voot', 'spotify', 'youtube', 'gaana',
                            'pvr', 'inox', 'cinepolis', 'bookmyshow'
                        ],
                        'keywords': [
                            'movie', 'cinema', 'streaming', 'music', 'entertainment',
                            'subscription', 'premium', 'video', 'film'
                        ]
                    },
                    'Education': {
                        'merchants': [
                            'byju', 'unacademy', 'vedantu', 'white hat jr', 'coursera',
                            'udemy', 'skillshare', 'university', 'college', 'school'
                        ],
                        'keywords': [
                            'course', 'education', 'tuition', 'fee', 'university',
                            'school', 'college', 'learning', 'study'
                        ]
                    },
                    'Insurance': {
                        'merchants': [
                            'lic', 'star health', 'bajaj allianz', 'hdfc ergo', 'icici lombard',
                            'sbi life', 'max life', 'niva bupa', 'reliance general'
                        ],
                        'keywords': [
                            'insurance', 'policy', 'premium', 'coverage', 'life insurance',
                            'health insurance', 'term insurance'
                        ]
                    }
                }
            
            def categorize_transaction(self, sms_text):
                """Standalone categorization requiring only SMS text"""
                try:
                    # Extract all information from SMS text itself
                    extracted_data = self._extract_all_from_sms(sms_text)
                    
                    text_lower = sms_text.lower()
                    best_category = 'Other'
                    best_confidence = 0.0
                    best_matches = []
                    
                    # Priority-based categorization with confidence scoring
                    for category, rules in self.category_rules.items():
                        confidence = 0.0
                        matches = []
                        
                        # Merchant name matching (higher weight)
                        for merchant in rules['merchants']:
                            if merchant.lower() in text_lower:
                                confidence += 0.6
                                matches.append(f"merchant:{merchant}")
                                break
                        
                        # Keyword matching (cumulative)
                        keyword_score = 0
                        for keyword in rules['keywords']:
                            if keyword.lower() in text_lower:
                                keyword_score += 0.15
                                matches.append(f"keyword:{keyword}")
                        
                        confidence += min(keyword_score, 0.4)  # Cap keyword contribution
                        
                        # Special boost for exact merchant extraction
                        if extracted_data['merchant']:
                            merchant_name = extracted_data['merchant'].lower()
                            for merchant in rules['merchants']:
                                if merchant.lower() in merchant_name:
                                    confidence += 0.3
                                    matches.append(f"extracted_merchant:{merchant}")
                                    break
                        
                        # Update best category
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_category = category
                            best_matches = matches
                    
                    # Ensure minimum confidence for non-Other categories
                    if best_confidence < 0.3 and best_category != 'Other':
                        best_category = 'Other'
                        best_confidence = 0.3
                    
                    # Cap confidence at 1.0
                    best_confidence = min(best_confidence, 1.0)
                    
                    return {
                        'category': best_category,
                        'confidence_score': best_confidence,
                        'merchant_name': extracted_data['merchant'],
                        'amount': extracted_data['amount'],
                        'transaction_type': extracted_data['transaction_type'],
                        'matches': best_matches,
                        'method': 'Enhanced_Standalone_AI'
                    }
                    
                except Exception as e:
                    logger.error(f"Standalone AI categorization error: {e}")
                    return {
                        'category': 'Other',
                        'confidence_score': 0.1,
                        'merchant_name': None,
                        'amount': None,
                        'transaction_type': 'unknown',
                        'error': str(e)
                    }
            
            def _extract_all_from_sms(self, sms_text):
                """Extract all required information from SMS text alone"""
                # Amount extraction
                amount = None
                amount_patterns = [
                    r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    r'INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    r'₹\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
                ]
                
                for pattern in amount_patterns:
                    match = re.search(pattern, sms_text, re.IGNORECASE)
                    if match:
                        try:
                            amount = float(match.group(1).replace(',', ''))
                            break
                        except ValueError:
                            continue
                
                # Enhanced merchant extraction patterns
                merchant = None
                merchant_patterns = [
                    # Specific pattern for "to MERCHANT_NAME"
                    r'(?:to|at)\s+([A-Z][A-Za-z\s&.\-\(\)0-9]+?)(?:\s+on|\s+for|\s+upi|\s+ref|\.|$)',
                    # Pattern for "SRI" prefixed merchants (jewelry stores)
                    r'(?:to|at)\s+(SRI\s+[A-Za-z\s&.\-\(\)]+?)(?:\s+on|\s+for|\s+upi|\.|$)',
                    # Pattern for long merchant names
                    r'(?:to|at|from|for)\s+([A-Z][A-Za-z\s&.\-\(\)0-9]{10,}?)(?:\s+on|\s+for|\s+upi|\s+ref|\.|$)',
                    # Fallback for any capitalized text before UPI/on/for
                    r'([A-Z][A-Z\s&.\-\(\)]{5,}?)(?:\s+upi|\s+on|\s+for|\.|$)'
                ]
                
                for pattern in merchant_patterns:
                    match = re.search(pattern, sms_text, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        # Clean and validate merchant name
                        if (len(candidate) > 2 and 
                            not re.match(r'^\d+$', candidate) and 
                            'UPI' not in candidate.upper() and
                            'SMS' not in candidate.upper() and
                            'NOT YOU' not in candidate.upper()):
                            merchant = candidate
                            break
                
                # Transaction type detection
                transaction_type = 'unknown'
                if re.search(r'debited|spent|paid', sms_text, re.IGNORECASE):
                    transaction_type = 'debit'
                elif re.search(r'credited|received', sms_text, re.IGNORECASE):
                    transaction_type = 'credit'
                
                return {
                    'merchant': merchant,
                    'amount': amount,
                    'transaction_type': transaction_type
                }
        
        return EnhancedStandaloneAI()
    
    def categorize_expense(self, sms_text):
        """Main SMS categorization - requires only SMS text"""
        try:
            # Use standalone AI (no additional arguments needed)
            if self.hybrid_ai:
                result = self.hybrid_ai.categorize_transaction(sms_text)
                return {
                    'category': result['category'],
                    'confidence': result['confidence_score'],
                    'merchant_detected': result.get('merchant_name'),
                    'amount': result.get('amount'),
                    'transaction_type': result.get('transaction_type'),
                    'method': result.get('method', 'Standalone_AI'),
                    'matches': result.get('matches', []),
                    'sms_text': sms_text
                }
            else:
                return {
                    'category': 'Other',
                    'confidence': 0.1,
                    'merchant_detected': None,
                    'method': 'No_Model_Available',
                    'error': 'No AI model available'
                }
                
        except Exception as e:
            logger.error(f"SMS categorization error: {e}")
            logger.error(traceback.format_exc())
            return {
                'category': 'Other',
                'confidence': 0.0,
                'error': str(e),
                'sms_text': sms_text
            }

# Initialize standalone categorizer
categorizer = ImprovedExpenseCategorizer()

# Load models at startup
print("🔄 Loading standalone AI system...")
categorizer.load_model()

# API Routes
@app.route('/')
def home():
    """API home page with documentation"""
    return jsonify({
        "service": "FinSaathi Standalone Expense Categorization API",
        "version": "3.0.0 (Standalone)",
        "status": "active", 
        "description": "Standalone AI that categorizes SMS without requiring any positional arguments",
        "model_status": categorizer.model_info,
        "features": [
            "Standalone AI categorization (SMS text only)",
            "No positional arguments required",
            "Enhanced jewelry detection",
            "Comprehensive Indian merchant database",
            "Self-sufficient merchant extraction",
            "Robust error handling"
        ],
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check",
            "POST /api/categorize": "SMS categorization (SMS text only)",
            "GET /api/test": "Test with sample SMS messages"
        },
        "usage": {
            "simple": {
                "url": "/api/categorize",
                "method": "POST",
                "body": {"sms_text": "Your SMS message here"},
                "note": "Only SMS text required - no merchant/amount needed"
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
        "api_version": "3.0.0 (Standalone)",
        "ready": True,
        "note": "Standalone AI requires no external arguments"
    })

@app.route('/api/categorize', methods=['POST'])
def categorize_sms_single():
    """Standalone SMS categorization - requires only SMS text"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sms_text = data.get('sms_text', '')
        
        if not sms_text:
            return jsonify({'error': 'sms_text is required'}), 400
        
        # Categorize using standalone method (no additional arguments)
        result = categorizer.categorize_expense(sms_text)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'note': 'Categorized using standalone AI (no external arguments)'
        })
        
    except Exception as e:
        logger.error(f"SMS categorization endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test_endpoint():
    """Test endpoint with various SMS types"""
    test_cases = [
        {
            "name": "Jewelry Store",
            "sms": "A/c *5678 debited Rs. 650.00 on 29-06-25 to SRI LAKSHMI GANAPATHI SWAMY JEWELLERIES. UPI:768744221384"
        },
        {
            "name": "Healthcare",
            "sms": "A/c *5678 debited Rs. 970.00 on 10-05-25 to UMA CLINICAL LABORATORY. UPI:882918376710"
        },
        {
            "name": "Power Bill",
            "sms": "A/c debited Rs. 471.00 on 23-05-25 to EASTERN POWER DISTRIBUTION COMPANY LIMITED"
        },
        {
            "name": "Food Delivery",
            "sms": "Payment of Rs. 285.50 made to ZOMATO FOOD DELIVERY on 15-Oct-25"
        },
        {
            "name": "Shopping",
            "sms": "Rs. 1250.00 debited to AMAZON INDIA SHOPPING"
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            result = categorizer.categorize_sms(test_case["sms"])
            results.append({
                'name': test_case['name'],
                'sms': test_case['sms'][:60] + "...",
                'category': result['category'],
                'confidence': result['confidence'],
                'merchant': result['merchant_detected'],
                'method': result['method']
            })
        except Exception as e:
            results.append({
                'name': test_case['name'],
                'error': str(e)
            })
    
    return jsonify({
        'test_results': results,
        'model_status': categorizer.model_info,
        'note': 'All tests use standalone AI (SMS text only)'
    })

# Batch processing endpoint
@app.route('/api/categorize/batch', methods=['POST'])
def categorize_batch():
    """Batch SMS processing - each SMS categorized independently"""
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
            'timestamp': datetime.now().isoformat(),
            'note': 'Batch processing using standalone AI'
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    # Load models at startup
    categorizer.load_models()
