import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from enhanced_training_data import get_enhanced_training_data

class ImprovedExpenseCategorizer:
    """Enhanced expense categorizer with better accuracy"""
    
    def __init__(self):
        # Enhanced text vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            stop_words='english',
            lowercase=True,
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True,  # Use sublinear scaling
            strip_accents='ascii'
        )
        
        # Use RandomForest as main classifier (works well with mixed features)
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for better accuracy
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.is_trained = False
        self.categories = [
            'Food & Dining', 'Transportation', 'Shopping', 'Groceries',
            'Bills & Utilities', 'Healthcare', 'Entertainment', 'Education',
            'Housing', 'Travel', 'Insurance', 'Investment', 'Other'
        ]
        
    def create_enhanced_features(self, description, merchant, amount=None):
        """Create enhanced features for better categorization"""
        # Clean and combine text
        desc_clean = self.clean_text(description)
        merchant_clean = self.clean_text(merchant) if merchant else ""
        
        # Combine all text features
        combined_text = f"{desc_clean} {merchant_clean}".strip()
        
        # Add keyword-based features
        combined_text += self.extract_keyword_features(combined_text, merchant_clean, amount)
        
        return combined_text
    
    def clean_text(self, text):
        """Advanced text cleaning"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s&.-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keyword_features(self, text, merchant, amount):
        """Extract keyword-based features as text tokens"""
        features = []
        
        # Category-specific keywords
        category_keywords = {
            'food_dining': ['restaurant', 'food', 'cafe', 'coffee', 'pizza', 'burger', 'meal', 'dining', 'kitchen', 'delivery', 'takeout', 'breakfast', 'lunch', 'dinner', 'snack', 'beverage', 'drink', 'zomato', 'swiggy', 'dominos', 'mcdonald', 'kfc', 'starbucks'],
            'transportation': ['uber', 'ola', 'taxi', 'fuel', 'petrol', 'gas', 'transport', 'metro', 'bus', 'train', 'auto', 'cab', 'parking', 'toll', 'vehicle', 'bike', 'car', 'rapido', 'irctc'],
            'shopping': ['amazon', 'flipkart', 'myntra', 'shopping', 'store', 'mall', 'retail', 'clothes', 'clothing', 'fashion', 'shoes', 'electronics', 'gadget', 'mobile', 'laptop', 'ajio', 'nykaa', 'snapdeal'],
            'groceries': ['grocery', 'supermarket', 'vegetables', 'fruits', 'market', 'fresh', 'bazaar', 'mart', 'provisions', 'dairy', 'organic', 'bigbasket', 'grofers'],
            'utilities': ['electricity', 'water', 'gas', 'internet', 'broadband', 'mobile', 'phone', 'utility', 'bill', 'recharge', 'dth', 'cable', 'airtel', 'vodafone', 'jio'],
            'healthcare': ['hospital', 'doctor', 'medical', 'pharmacy', 'medicine', 'health', 'clinic', 'dental', 'eye', 'care', 'treatment', 'consultation', 'apollo', 'lenskart'],
            'entertainment': ['movie', 'cinema', 'netflix', 'entertainment', 'game', 'music', 'streaming', 'concert', 'show', 'park', 'fun', 'pvr', 'inox', 'spotify', 'amazon prime'],
            'education': ['education', 'school', 'university', 'course', 'learning', 'training', 'coaching', 'books', 'study', 'class', 'coursera', 'udemy'],
            'housing': ['rent', 'house', 'home', 'property', 'maintenance', 'repair', 'furniture', 'loan', 'emi', 'housing', 'urban company'],
            'travel': ['flight', 'hotel', 'travel', 'trip', 'vacation', 'airline', 'booking', 'tourist', 'journey', 'indigo', 'makemytrip'],
            'insurance': ['insurance', 'premium', 'policy', 'coverage', 'claim', 'lic', 'bajaj allianz'],
            'investment': ['investment', 'mutual', 'fund', 'stock', 'share', 'sip', 'deposit', 'gold', 'bond', 'zerodha', 'groww']
        }
        
        # Add category keyword indicators
        for category, keywords in category_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text)
            if keyword_count > 0:
                features.append(f"category_{category}")
                features.append(f"category_{category}_count_{min(keyword_count, 5)}")  # Cap at 5
        
        # Merchant-specific features
        if merchant:
            # Known merchant categories
            known_merchants = {
                'amazon': 'shopping_online',
                'flipkart': 'shopping_online',
                'myntra': 'shopping_fashion',
                'zomato': 'food_delivery',
                'swiggy': 'food_delivery',
                'uber': 'transport_cab',
                'ola': 'transport_cab',
                'netflix': 'entertainment_streaming',
                'spotify': 'entertainment_music',
                'airtel': 'utility_telecom',
                'vodafone': 'utility_telecom',
                'apollo': 'healthcare_hospital'
            }
            
            for merchant_key, merchant_type in known_merchants.items():
                if merchant_key in merchant:
                    features.append(f"merchant_type_{merchant_type}")
        
        # Amount-based features
        if amount is not None:
            if amount < 50:
                features.append("amount_very_small")
            elif amount < 200:
                features.append("amount_small")
            elif amount < 1000:
                features.append("amount_medium")
            elif amount < 5000:
                features.append("amount_large")
            else:
                features.append("amount_very_large")
            
            # Typical amount ranges for categories
            if 100 <= amount <= 1000:
                features.append("subscription_range")
            if 200 <= amount <= 3000:
                features.append("utility_range")
            if 500 <= amount <= 5000:
                features.append("grocery_range")
            if amount > 10000:
                features.append("major_purchase")
        
        return " " + " ".join(features) if features else ""
    
    def train_model(self, training_data=None):
        """Train the improved model"""
        try:
            if training_data is None:
                training_data = get_enhanced_training_data()
                
            print(f"Training with {len(training_data)} samples...")
            
            # Prepare features
            features = []
            labels = []
            
            for item in training_data:
                feature_text = self.create_enhanced_features(
                    item.get('description', ''),
                    item.get('merchant', ''),
                    item.get('amount')
                )
                features.append(feature_text)
                labels.append(item['category'])
            
            # Train vectorizer and model
            X = self.vectorizer.fit_transform(features)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train model
            print("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Validation accuracy: {test_score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, labels, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Detailed classification report
            y_pred = self.model.predict(X_test)
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Feature importance
            feature_names = self.vectorizer.get_feature_names_out()
            feature_importance = self.model.feature_importances_
            top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:20]
            
            print("\nTop 20 Most Important Features:")
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.4f}")
            
            self.is_trained = True
            self.save_model()
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def categorize_expense(self, description, merchant, amount=None):
        """Categorize expense with improved accuracy"""
        if not self.is_trained:
            if not self.load_model():
                print("Model not trained. Training now...")
                if not self.train_model():
                    return {'error': 'Failed to train model'}
        
        try:
            # Create features
            feature_text = self.create_enhanced_features(description, merchant, amount)
            X = self.vectorizer.transform([feature_text])
            
            # Predict
            category = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get confidence score
            confidence = np.max(probabilities)
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            predictions = [
                {
                    'category': self.model.classes_[i],
                    'confidence': float(probabilities[i])
                }
                for i in top_indices
            ]
            
            return {
                'primary_category': category,
                'confidence': float(confidence),
                'all_predictions': predictions,
                'model_type': 'random_forest_enhanced'
            }
            
        except Exception as e:
            print(f"Categorization error: {e}")
            return {'error': str(e)}
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'is_trained': self.is_trained,
                'categories': self.categories
            }
            
            with open('enhanced_expense_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print("Enhanced model saved successfully!")
            
        except Exception as e:
            print("Error saving model:", e)
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('enhanced_expense_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.is_trained = model_data['is_trained']
            self.categories = model_data['categories']
            
            print("Enhanced model loaded successfully!")
            return True
            
        except Exception as e:
            print("Error loading model:", e)
            return False

def test_enhanced_model():
    """Test the enhanced model with sample data"""
    categorizer = ImprovedExpenseCategorizer()
    
    # Train the model
    print("Training enhanced model...")
    success = categorizer.train_model()
    
    if not success:
        print("Failed to train model")
        return
    
    # Test cases
    test_cases = [
        {
            'description': 'UPI payment to MYNTRA FASHION STORE for clothing purchase',
            'merchant': 'MYNTRA FASHION STORE',
            'amount': 2500.00
        },
        {
            'description': 'Zomato food delivery order dinner',
            'merchant': 'ZOMATO',
            'amount': 350.00
        },
        {
            'description': 'Netflix subscription monthly streaming service',
            'merchant': 'NETFLIX',
            'amount': 649.00
        },
        {
            'description': 'Big Bazaar grocery shopping weekly essentials',
            'merchant': 'BIG BAZAAR',
            'amount': 2200.00
        },
        {
            'description': 'Uber cab ride to office transportation',
            'merchant': 'UBER',
            'amount': 180.00
        },
        {
            'description': 'Apollo Hospital doctor consultation fee',
            'merchant': 'APOLLO HOSPITAL',
            'amount': 800.00
        },
        {
            'description': 'Airtel broadband internet bill payment',
            'merchant': 'AIRTEL BROADBAND',
            'amount': 999.00
        },
        {
            'description': 'Flipkart electronics smartphone purchase',
            'merchant': 'FLIPKART',
            'amount': 15000.00
        }
    ]
    
    print("\nTesting enhanced model:")
    print("=" * 60)
    
    total_tests = len(test_cases)
    high_confidence = 0
    
    for i, test in enumerate(test_cases, 1):
        result = categorizer.categorize_expense(
            test['description'],
            test['merchant'],
            test['amount']
        )
        
        print(f"\n{i}. {test['description']}")
        print(f"   Merchant: {test['merchant']}")
        print(f"   Amount: Rs.{test['amount']}")
        print(f"   Predicted: {result['primary_category']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
        if result['confidence'] > 0.7:
            high_confidence += 1
            
        print("   Top 3 predictions:")
        for pred in result['all_predictions']:
            print(f"     - {pred['category']}: {pred['confidence']:.1%}")
    
    print("\nModel Performance Summary:")
    print(f"- Tests with >70% confidence: {high_confidence}/{total_tests} ({high_confidence/total_tests*100:.1f}%)")
    print(f"- Model type: {result.get('model_type', 'unknown')}")

if __name__ == '__main__':
    test_enhanced_model()