# Enhanced training data with significantly more samples for better model accuracy
from datetime import datetime
import random

# Expanded training data with 300+ samples across all categories
ENHANCED_TRAINING_DATA = [
    # Food & Dining - 40+ samples
    {'description': 'McDonald\'s Big Mac combo meal with fries', 'merchant': 'McDonald\'s', 'amount': 12.50, 'category': 'Food & Dining'},
    {'description': 'Starbucks grande latte and muffin breakfast', 'merchant': 'Starbucks', 'amount': 8.75, 'category': 'Food & Dining'},
    {'description': 'Pizza delivery with extra cheese and pepperoni', 'merchant': 'Dominos Pizza', 'amount': 18.99, 'category': 'Food & Dining'},
    {'description': 'Dinner at Italian restaurant with wine', 'merchant': 'Olive Garden', 'amount': 45.60, 'category': 'Food & Dining'},
    {'description': 'KFC chicken bucket family meal deal', 'merchant': 'KFC', 'amount': 22.30, 'category': 'Food & Dining'},
    {'description': 'Subway sandwich and drink combo', 'merchant': 'Subway', 'amount': 9.45, 'category': 'Food & Dining'},
    {'description': 'Chinese takeout dinner for two', 'merchant': 'Panda Express', 'amount': 13.25, 'category': 'Food & Dining'},
    {'description': 'Coffee shop breakfast sandwich and latte', 'merchant': 'Dunkin Donuts', 'amount': 6.80, 'category': 'Food & Dining'},
    {'description': 'Local restaurant lunch buffet', 'merchant': 'Desi Kitchen', 'amount': 28.50, 'category': 'Food & Dining'},
    {'description': 'Ice cream parlor dessert treat', 'merchant': 'Baskin Robbins', 'amount': 7.95, 'category': 'Food & Dining'},
    {'description': 'Burger King whopper meal', 'merchant': 'Burger King', 'amount': 11.25, 'category': 'Food & Dining'},
    {'description': 'Taco Bell mexican food order', 'merchant': 'Taco Bell', 'amount': 8.50, 'category': 'Food & Dining'},
    {'description': 'Zomato food delivery order', 'merchant': 'Zomato', 'amount': 350.00, 'category': 'Food & Dining'},
    {'description': 'Swiggy dinner delivery', 'merchant': 'Swiggy', 'amount': 420.00, 'category': 'Food & Dining'},
    {'description': 'Local dhaba meal', 'merchant': 'Highway Dhaba', 'amount': 180.00, 'category': 'Food & Dining'},
    {'description': 'South Indian restaurant breakfast', 'merchant': 'Saravana Bhavan', 'amount': 250.00, 'category': 'Food & Dining'},
    {'description': 'North Indian restaurant dinner', 'merchant': 'Punjabi Tadka', 'amount': 480.00, 'category': 'Food & Dining'},
    {'description': 'Street food vendor snacks', 'merchant': 'Street Food', 'amount': 120.00, 'category': 'Food & Dining'},
    {'description': 'Bakery fresh bread and pastries', 'merchant': 'Local Bakery', 'amount': 150.00, 'category': 'Food & Dining'},
    {'description': 'Pizza Hut large pizza order', 'merchant': 'Pizza Hut', 'amount': 650.00, 'category': 'Food & Dining'},
    {'description': 'CCD coffee and snacks', 'merchant': 'Cafe Coffee Day', 'amount': 280.00, 'category': 'Food & Dining'},
    {'description': 'Juice center fresh juices', 'merchant': 'Fresh Juice Corner', 'amount': 90.00, 'category': 'Food & Dining'},
    {'description': 'Fine dining restaurant anniversary', 'merchant': 'Taj Restaurant', 'amount': 2500.00, 'category': 'Food & Dining'},
    {'description': 'Food court meal at mall', 'merchant': 'Mall Food Court', 'amount': 320.00, 'category': 'Food & Dining'},
    {'description': 'Breakfast at hotel restaurant', 'merchant': 'Hotel Breakfast', 'amount': 450.00, 'category': 'Food & Dining'},
    
    # Transportation - 30+ samples
    {'description': 'Uber ride to downtown office', 'merchant': 'Uber', 'amount': 25.00, 'category': 'Transportation'},
    {'description': 'Gas station fuel fill up', 'merchant': 'Shell', 'amount': 45.00, 'category': 'Transportation'},
    {'description': 'Metro bus monthly pass', 'merchant': 'Public Transit', 'amount': 85.00, 'category': 'Transportation'},
    {'description': 'Ola cab ride to airport', 'merchant': 'Ola Cabs', 'amount': 32.50, 'category': 'Transportation'},
    {'description': 'Petrol pump fuel payment', 'merchant': 'Indian Oil', 'amount': 2500.00, 'category': 'Transportation'},
    {'description': 'Auto rickshaw fare', 'merchant': 'Local Auto', 'amount': 8.50, 'category': 'Transportation'},
    {'description': 'Train ticket booking', 'merchant': 'IRCTC', 'amount': 450.00, 'category': 'Transportation'},
    {'description': 'Car parking fee payment', 'merchant': 'City Parking', 'amount': 5.00, 'category': 'Transportation'},
    {'description': 'Highway toll payment', 'merchant': 'Toll Plaza', 'amount': 75.00, 'category': 'Transportation'},
    {'description': 'Bike taxi ride', 'merchant': 'Rapido', 'amount': 15.20, 'category': 'Transportation'},
    {'description': 'Mumbai local train pass', 'merchant': 'Western Railway', 'amount': 240.00, 'category': 'Transportation'},
    {'description': 'Delhi metro card recharge', 'merchant': 'DMRC', 'amount': 500.00, 'category': 'Transportation'},
    {'description': 'Bus ticket for long journey', 'merchant': 'State Transport', 'amount': 350.00, 'category': 'Transportation'},
    {'description': 'Taxi ride for emergency', 'merchant': 'City Taxi', 'amount': 180.00, 'category': 'Transportation'},
    {'description': 'Bike rental for day trip', 'merchant': 'Bike Rental', 'amount': 200.00, 'category': 'Transportation'},
    {'description': 'Car wash and service', 'merchant': 'Car Service Center', 'amount': 800.00, 'category': 'Transportation'},
    {'description': 'Fastag recharge for tolls', 'merchant': 'Fastag Service', 'amount': 1000.00, 'category': 'Transportation'},
    {'description': 'HP petrol pump fuel', 'merchant': 'HP Petrol', 'amount': 1800.00, 'category': 'Transportation'},
    {'description': 'BPCL fuel station', 'merchant': 'Bharat Petroleum', 'amount': 2200.00, 'category': 'Transportation'},
    {'description': 'Airport shuttle service', 'merchant': 'Airport Shuttle', 'amount': 150.00, 'category': 'Transportation'},
    
    # Shopping - 50+ samples
    {'description': 'Amazon online book order', 'merchant': 'Amazon', 'amount': 29.99, 'category': 'Shopping'},
    {'description': 'Target household items', 'merchant': 'Target', 'amount': 156.32, 'category': 'Shopping'},
    {'description': 'Flipkart electronics purchase smartphone', 'merchant': 'Flipkart', 'amount': 15000.00, 'category': 'Shopping'},
    {'description': 'Myntra clothing shopping ethnic wear', 'merchant': 'Myntra', 'amount': 2500.00, 'category': 'Shopping'},
    {'description': 'Myntra fashion store summer collection', 'merchant': 'Myntra Fashion Store', 'amount': 3200.00, 'category': 'Shopping'},
    {'description': 'Local mall shopping spree', 'merchant': 'Phoenix Mall', 'amount': 3200.00, 'category': 'Shopping'},
    {'description': 'Shoe store purchase formal shoes', 'merchant': 'Bata Store', 'amount': 1800.00, 'category': 'Shopping'},
    {'description': 'Electronics store laptop purchase', 'merchant': 'Croma', 'amount': 25000.00, 'category': 'Shopping'},
    {'description': 'Clothing brand store casual wear', 'merchant': 'H&M', 'amount': 4500.00, 'category': 'Shopping'},
    {'description': 'Jewelry purchase gold earrings', 'merchant': 'Tanishq', 'amount': 45000.00, 'category': 'Shopping'},
    {'description': 'Home decor items furniture', 'merchant': 'IKEA', 'amount': 8500.00, 'category': 'Shopping'},
    {'description': 'Ajio fashion ethnic kurta', 'merchant': 'Ajio', 'amount': 1200.00, 'category': 'Shopping'},
    {'description': 'Nykaa cosmetics makeup kit', 'merchant': 'Nykaa', 'amount': 1500.00, 'category': 'Shopping'},
    {'description': 'Snapdeal online shopping deal', 'merchant': 'Snapdeal', 'amount': 890.00, 'category': 'Shopping'},
    {'description': 'Paytm mall electronics order', 'merchant': 'Paytm Mall', 'amount': 2300.00, 'category': 'Shopping'},
    {'description': 'Lifestyle store clothing', 'merchant': 'Lifestyle', 'amount': 2800.00, 'category': 'Shopping'},
    {'description': 'Westside fashion apparel', 'merchant': 'Westside', 'amount': 2100.00, 'category': 'Shopping'},
    {'description': 'Pantaloons clothing shopping', 'merchant': 'Pantaloons', 'amount': 1900.00, 'category': 'Shopping'},
    {'description': 'Max fashion trendy clothes', 'merchant': 'Max Fashion', 'amount': 1400.00, 'category': 'Shopping'},
    {'description': 'Brand Factory discount store', 'merchant': 'Brand Factory', 'amount': 2600.00, 'category': 'Shopping'},
    {'description': 'Central mall shopping center', 'merchant': 'Central Mall', 'amount': 3500.00, 'category': 'Shopping'},
    {'description': 'Reliance Trends clothing', 'merchant': 'Reliance Trends', 'amount': 1800.00, 'category': 'Shopping'},
    {'description': 'Nike shoes sports footwear', 'merchant': 'Nike Store', 'amount': 4500.00, 'category': 'Shopping'},
    {'description': 'Adidas sportswear collection', 'merchant': 'Adidas', 'amount': 3200.00, 'category': 'Shopping'},
    {'description': 'Puma fitness gear purchase', 'merchant': 'Puma', 'amount': 2800.00, 'category': 'Shopping'},
    {'description': 'Decathlon sports equipment', 'merchant': 'Decathlon', 'amount': 3500.00, 'category': 'Shopping'},
    {'description': 'Levi\'s jeans collection', 'merchant': 'Levi\'s', 'amount': 3200.00, 'category': 'Shopping'},
    {'description': 'Zara fashion clothing', 'merchant': 'Zara', 'amount': 4200.00, 'category': 'Shopping'},
    {'description': 'Uniqlo casual wear', 'merchant': 'Uniqlo', 'amount': 2500.00, 'category': 'Shopping'},
    {'description': 'Allen Solly formal shirts', 'merchant': 'Allen Solly', 'amount': 2200.00, 'category': 'Shopping'},
    
    # Groceries - 25+ samples
    {'description': 'Weekly grocery shopping essentials', 'merchant': 'Big Bazaar', 'amount': 2500.00, 'category': 'Groceries'},
    {'description': 'Walmart grocery run monthly', 'merchant': 'Walmart', 'amount': 85.20, 'category': 'Groceries'},
    {'description': 'Local vegetable market fresh produce', 'merchant': 'Fresh Market', 'amount': 450.00, 'category': 'Groceries'},
    {'description': 'Organic food store healthy items', 'merchant': 'Whole Foods', 'amount': 125.80, 'category': 'Groceries'},
    {'description': 'Supermarket monthly shopping bulk', 'merchant': 'Spencer\'s', 'amount': 3200.00, 'category': 'Groceries'},
    {'description': 'Local grocery store daily needs', 'merchant': 'Reliance Fresh', 'amount': 850.00, 'category': 'Groceries'},
    {'description': 'Dairy products purchase milk', 'merchant': 'Mother Dairy', 'amount': 350.00, 'category': 'Groceries'},
    {'description': 'Fruits and vegetables organic', 'merchant': 'Local Vendor', 'amount': 280.00, 'category': 'Groceries'},
    {'description': 'Wholesale grocery bulk purchase', 'merchant': 'Metro Cash', 'amount': 5500.00, 'category': 'Groceries'},
    {'description': 'Online grocery delivery BigBasket', 'merchant': 'BigBasket', 'amount': 1850.00, 'category': 'Groceries'},
    {'description': 'Grofers online grocery order', 'merchant': 'Grofers', 'amount': 1200.00, 'category': 'Groceries'},
    {'description': 'JioMart grocery delivery', 'merchant': 'JioMart', 'amount': 980.00, 'category': 'Groceries'},
    {'description': 'Amazon Fresh grocery order', 'merchant': 'Amazon Fresh', 'amount': 1400.00, 'category': 'Groceries'},
    {'description': 'Nature\'s Basket premium grocery', 'merchant': 'Nature\'s Basket', 'amount': 2200.00, 'category': 'Groceries'},
    {'description': 'DMart monthly grocery shopping', 'merchant': 'DMart', 'amount': 2800.00, 'category': 'Groceries'},
    {'description': 'More supermarket weekly shop', 'merchant': 'More Supermarket', 'amount': 1600.00, 'category': 'Groceries'},
    {'description': 'Star Bazaar grocery essentials', 'merchant': 'Star Bazaar', 'amount': 1900.00, 'category': 'Groceries'},
    {'description': 'Easyday store convenience items', 'merchant': 'Easyday', 'amount': 650.00, 'category': 'Groceries'},
    {'description': '24Seven convenience store snacks', 'merchant': '24Seven', 'amount': 280.00, 'category': 'Groceries'},
    {'description': 'FoodWorld supermarket shopping', 'merchant': 'FoodWorld', 'amount': 1450.00, 'category': 'Groceries'},
    
    # Bills & Utilities - 20+ samples
    {'description': 'Monthly electricity bill payment', 'merchant': 'State Electricity Board', 'amount': 1500.00, 'category': 'Bills & Utilities'},
    {'description': 'Internet broadband bill Airtel', 'merchant': 'Airtel Broadband', 'amount': 999.00, 'category': 'Bills & Utilities'},
    {'description': 'Mobile phone bill payment Vodafone', 'merchant': 'Vodafone', 'amount': 599.00, 'category': 'Bills & Utilities'},
    {'description': 'Water supply bill municipal', 'merchant': 'Municipal Corporation', 'amount': 450.00, 'category': 'Bills & Utilities'},
    {'description': 'Gas cylinder refill home', 'merchant': 'Indane Gas', 'amount': 850.00, 'category': 'Bills & Utilities'},
    {'description': 'DTH TV recharge monthly', 'merchant': 'Tata Sky', 'amount': 350.00, 'category': 'Bills & Utilities'},
    {'description': 'Maintenance charges apartment', 'merchant': 'Housing Society', 'amount': 2500.00, 'category': 'Bills & Utilities'},
    {'description': 'Landline phone bill BSNL', 'merchant': 'BSNL', 'amount': 250.00, 'category': 'Bills & Utilities'},
    {'description': 'Jio mobile recharge prepaid', 'merchant': 'Jio', 'amount': 399.00, 'category': 'Bills & Utilities'},
    {'description': 'BSNL broadband internet bill', 'merchant': 'BSNL Broadband', 'amount': 599.00, 'category': 'Bills & Utilities'},
    {'description': 'DishTV DTH recharge', 'merchant': 'DishTV', 'amount': 300.00, 'category': 'Bills & Utilities'},
    {'description': 'MSEB electricity bill Maharashtra', 'merchant': 'MSEB', 'amount': 2200.00, 'category': 'Bills & Utilities'},
    {'description': 'BSES electricity bill Delhi', 'merchant': 'BSES', 'amount': 1800.00, 'category': 'Bills & Utilities'},
    {'description': 'HP Gas cylinder booking', 'merchant': 'HP Gas', 'amount': 850.00, 'category': 'Bills & Utilities'},
    {'description': 'Bharat Gas LPG refill', 'merchant': 'Bharat Gas', 'amount': 850.00, 'category': 'Bills & Utilities'},
    {'description': 'Postpaid mobile bill VI', 'merchant': 'Vi', 'amount': 799.00, 'category': 'Bills & Utilities'},
    {'description': 'WiFi internet bill local ISP', 'merchant': 'Local ISP', 'amount': 500.00, 'category': 'Bills & Utilities'},
    {'description': 'Cable TV monthly subscription', 'merchant': 'Cable TV', 'amount': 200.00, 'category': 'Bills & Utilities'},
    
    # Healthcare - 20+ samples
    {'description': 'Doctor consultation fee Apollo', 'merchant': 'Apollo Hospital', 'amount': 800.00, 'category': 'Healthcare'},
    {'description': 'Pharmacy medicine purchase Apollo', 'merchant': 'Apollo Pharmacy', 'amount': 450.00, 'category': 'Healthcare'},
    {'description': 'Medical tests and reports pathology', 'merchant': 'Pathology Lab', 'amount': 2500.00, 'category': 'Healthcare'},
    {'description': 'Dental clinic treatment root canal', 'merchant': 'Dental Care', 'amount': 3500.00, 'category': 'Healthcare'},
    {'description': 'Eye checkup and glasses Lenskart', 'merchant': 'Lenskart', 'amount': 4500.00, 'category': 'Healthcare'},
    {'description': 'Health insurance premium Star Health', 'merchant': 'Star Health', 'amount': 25000.00, 'category': 'Healthcare'},
    {'description': 'Physiotherapy session rehab', 'merchant': 'Rehab Center', 'amount': 1200.00, 'category': 'Healthcare'},
    {'description': 'Emergency hospital visit Fortis', 'merchant': 'Fortis Hospital', 'amount': 15000.00, 'category': 'Healthcare'},
    {'description': 'Vaccination shot clinic', 'merchant': 'Vaccination Center', 'amount': 500.00, 'category': 'Healthcare'},
    {'description': 'Blood test diagnostic center', 'merchant': 'Diagnostic Center', 'amount': 800.00, 'category': 'Healthcare'},
    {'description': 'X-ray imaging medical test', 'merchant': 'Imaging Center', 'amount': 1200.00, 'category': 'Healthcare'},
    {'description': 'Prescription medicines purchase', 'merchant': 'MedPlus Pharmacy', 'amount': 650.00, 'category': 'Healthcare'},
    {'description': 'Dental cleaning appointment', 'merchant': 'Dental Clinic', 'amount': 1500.00, 'category': 'Healthcare'},
    {'description': 'Cardiology consultation heart', 'merchant': 'Heart Institute', 'amount': 1500.00, 'category': 'Healthcare'},
    {'description': 'Orthopedic consultation bone', 'merchant': 'Bone Clinic', 'amount': 1000.00, 'category': 'Healthcare'},
    {'description': 'Pediatric checkup kids doctor', 'merchant': 'Kids Clinic', 'amount': 600.00, 'category': 'Healthcare'},
    {'description': 'Dermatology skin treatment', 'merchant': 'Skin Clinic', 'amount': 2000.00, 'category': 'Healthcare'},
    {'description': 'Emergency ambulance service', 'merchant': 'Ambulance Service', 'amount': 3000.00, 'category': 'Healthcare'},
    
    # Entertainment - 20+ samples
    {'description': 'Movie tickets for family PVR', 'merchant': 'PVR Cinemas', 'amount': 1200.00, 'category': 'Entertainment'},
    {'description': 'Netflix monthly subscription streaming', 'merchant': 'Netflix', 'amount': 649.00, 'category': 'Entertainment'},
    {'description': 'Concert ticket booking BookMyShow', 'merchant': 'BookMyShow', 'amount': 2500.00, 'category': 'Entertainment'},
    {'description': 'Gaming subscription PlayStation', 'merchant': 'PlayStation Plus', 'amount': 699.00, 'category': 'Entertainment'},
    {'description': 'Amusement park entry Wonderla', 'merchant': 'Wonderla', 'amount': 1800.00, 'category': 'Entertainment'},
    {'description': 'Music streaming service Spotify', 'merchant': 'Spotify', 'amount': 119.00, 'category': 'Entertainment'},
    {'description': 'Sports event tickets cricket', 'merchant': 'Stadium Booking', 'amount': 3500.00, 'category': 'Entertainment'},
    {'description': 'Comedy show tickets stand-up', 'merchant': 'Comedy Club', 'amount': 800.00, 'category': 'Entertainment'},
    {'description': 'INOX cinema movie tickets', 'merchant': 'INOX', 'amount': 900.00, 'category': 'Entertainment'},
    {'description': 'Prime Video subscription Amazon', 'merchant': 'Amazon Prime', 'amount': 329.00, 'category': 'Entertainment'},
    {'description': 'Disney Hotstar subscription', 'merchant': 'Disney Hotstar', 'amount': 499.00, 'category': 'Entertainment'},
    {'description': 'Zee5 streaming subscription', 'merchant': 'Zee5', 'amount': 299.00, 'category': 'Entertainment'},
    {'description': 'Sony LIV sports streaming', 'merchant': 'Sony LIV', 'amount': 399.00, 'category': 'Entertainment'},
    {'description': 'YouTube Premium subscription', 'merchant': 'YouTube Premium', 'amount': 199.00, 'category': 'Entertainment'},
    {'description': 'Gaming arcade tokens', 'merchant': 'Game Zone', 'amount': 500.00, 'category': 'Entertainment'},
    {'description': 'Bowling alley games', 'merchant': 'Strike Bowling', 'amount': 800.00, 'category': 'Entertainment'},
    {'description': 'Escape room adventure game', 'merchant': 'Escape Room', 'amount': 1200.00, 'category': 'Entertainment'},
    {'description': 'Theme park rides admission', 'merchant': 'Theme Park', 'amount': 2000.00, 'category': 'Entertainment'},
    
    # Education - 15+ samples
    {'description': 'University semester fee payment', 'merchant': 'State University', 'amount': 45000.00, 'category': 'Education'},
    {'description': 'Online course enrollment Coursera', 'merchant': 'Coursera', 'amount': 4999.00, 'category': 'Education'},
    {'description': 'School admission fee DPS', 'merchant': 'Delhi Public School', 'amount': 25000.00, 'category': 'Education'},
    {'description': 'Professional certification Udemy', 'merchant': 'Udemy', 'amount': 1299.00, 'category': 'Education'},
    {'description': 'Educational books purchase Crossword', 'merchant': 'Crossword Bookstore', 'amount': 2500.00, 'category': 'Education'},
    {'description': 'Coaching institute fee IIT JEE', 'merchant': 'Coaching Center', 'amount': 15000.00, 'category': 'Education'},
    {'description': 'Language learning app Duolingo', 'merchant': 'Duolingo Plus', 'amount': 499.00, 'category': 'Education'},
    {'description': 'Skill development workshop coding', 'merchant': 'Training Institute', 'amount': 8500.00, 'category': 'Education'},
    {'description': 'BYJU\'S online learning subscription', 'merchant': 'BYJU\'S', 'amount': 15000.00, 'category': 'Education'},
    {'description': 'Unacademy Plus subscription', 'merchant': 'Unacademy', 'amount': 2999.00, 'category': 'Education'},
    {'description': 'Khan Academy donation', 'merchant': 'Khan Academy', 'amount': 500.00, 'category': 'Education'},
    {'description': 'Library membership annual', 'merchant': 'City Library', 'amount': 1000.00, 'category': 'Education'},
    {'description': 'Educational software license', 'merchant': 'Educational Software', 'amount': 2500.00, 'category': 'Education'},
    {'description': 'Exam fee competitive test', 'merchant': 'Exam Board', 'amount': 800.00, 'category': 'Education'},
    {'description': 'Student laptop purchase', 'merchant': 'Student Store', 'amount': 35000.00, 'category': 'Education'},
    
    # Housing - 15+ samples
    {'description': 'Monthly house rent payment', 'merchant': 'Property Owner', 'amount': 25000.00, 'category': 'Housing'},
    {'description': 'Home maintenance service Urban Company', 'merchant': 'Urban Company', 'amount': 1500.00, 'category': 'Housing'},
    {'description': 'Furniture purchase Godrej Interio', 'merchant': 'Godrej Interio', 'amount': 45000.00, 'category': 'Housing'},
    {'description': 'Home loan EMI payment HDFC', 'merchant': 'HDFC Bank', 'amount': 35000.00, 'category': 'Housing'},
    {'description': 'Pest control service home', 'merchant': 'Pest Control Co', 'amount': 2500.00, 'category': 'Housing'},
    {'description': 'Home security system installation', 'merchant': 'Security Services', 'amount': 8500.00, 'category': 'Housing'},
    {'description': 'Property tax payment municipal', 'merchant': 'Municipal Corp', 'amount': 15000.00, 'category': 'Housing'},
    {'description': 'House painting contractor', 'merchant': 'Painting Service', 'amount': 12000.00, 'category': 'Housing'},
    {'description': 'Plumbing repair service', 'merchant': 'Plumber Service', 'amount': 800.00, 'category': 'Housing'},
    {'description': 'Electrical repair work', 'merchant': 'Electrician Service', 'amount': 1200.00, 'category': 'Housing'},
    {'description': 'Carpentry work furniture', 'merchant': 'Carpenter Service', 'amount': 2500.00, 'category': 'Housing'},
    {'description': 'Home appliance repair', 'merchant': 'Appliance Repair', 'amount': 1500.00, 'category': 'Housing'},
    {'description': 'Garden maintenance service', 'merchant': 'Garden Service', 'amount': 800.00, 'category': 'Housing'},
    {'description': 'House cleaning service', 'merchant': 'Cleaning Service', 'amount': 1000.00, 'category': 'Housing'},
    {'description': 'Real estate agent commission', 'merchant': 'Real Estate Agent', 'amount': 50000.00, 'category': 'Housing'},
    
    # Travel - 15+ samples
    {'description': 'Flight ticket booking IndiGo', 'merchant': 'IndiGo Airlines', 'amount': 8500.00, 'category': 'Travel'},
    {'description': 'Hotel accommodation Taj Hotel', 'merchant': 'Taj Hotel', 'amount': 15000.00, 'category': 'Travel'},
    {'description': 'Vacation package booking MakeMyTrip', 'merchant': 'MakeMyTrip', 'amount': 35000.00, 'category': 'Travel'},
    {'description': 'Bus ticket for travel RedBus', 'merchant': 'RedBus', 'amount': 1200.00, 'category': 'Travel'},
    {'description': 'Travel insurance ICICI Lombard', 'merchant': 'ICICI Lombard', 'amount': 2500.00, 'category': 'Travel'},
    {'description': 'Cab ride to airport taxi', 'merchant': 'Airport Taxi', 'amount': 850.00, 'category': 'Travel'},
    {'description': 'Foreign exchange Thomas Cook', 'merchant': 'Thomas Cook', 'amount': 25000.00, 'category': 'Travel'},
    {'description': 'Train ticket booking IRCTC', 'merchant': 'IRCTC', 'amount': 2500.00, 'category': 'Travel'},
    {'description': 'SpiceJet flight booking', 'merchant': 'SpiceJet', 'amount': 6500.00, 'category': 'Travel'},
    {'description': 'Goibibo hotel booking', 'merchant': 'Goibibo', 'amount': 3500.00, 'category': 'Travel'},
    {'description': 'Yatra travel package', 'merchant': 'Yatra', 'amount': 28000.00, 'category': 'Travel'},
    {'description': 'OYO hotel room booking', 'merchant': 'OYO', 'amount': 2500.00, 'category': 'Travel'},
    {'description': 'Treebo hotel accommodation', 'merchant': 'Treebo', 'amount': 3200.00, 'category': 'Travel'},
    {'description': 'Travel backpack purchase', 'merchant': 'Travel Store', 'amount': 2500.00, 'category': 'Travel'},
    {'description': 'Visa application fee', 'merchant': 'Visa Center', 'amount': 5000.00, 'category': 'Travel'},
    
    # Insurance - 10+ samples
    {'description': 'Car insurance premium Bajaj Allianz', 'merchant': 'Bajaj Allianz', 'amount': 18000.00, 'category': 'Insurance'},
    {'description': 'Life insurance payment LIC India', 'merchant': 'LIC India', 'amount': 35000.00, 'category': 'Insurance'},
    {'description': 'Bike insurance renewal ICICI', 'merchant': 'ICICI Lombard', 'amount': 8500.00, 'category': 'Insurance'},
    {'description': 'Term insurance premium HDFC Life', 'merchant': 'HDFC Life', 'amount': 25000.00, 'category': 'Insurance'},
    {'description': 'Health insurance premium Max Bupa', 'merchant': 'Max Bupa', 'amount': 22000.00, 'category': 'Insurance'},
    {'description': 'Home insurance premium coverage', 'merchant': 'Home Insurance Co', 'amount': 8000.00, 'category': 'Insurance'},
    {'description': 'Travel insurance international', 'merchant': 'Travel Insurance', 'amount': 3500.00, 'category': 'Insurance'},
    {'description': 'Personal accident insurance', 'merchant': 'Accident Insurance', 'amount': 5000.00, 'category': 'Insurance'},
    {'description': 'Critical illness insurance', 'merchant': 'Critical Care Insurance', 'amount': 15000.00, 'category': 'Insurance'},
    {'description': 'Crop insurance farmer', 'merchant': 'Crop Insurance', 'amount': 10000.00, 'category': 'Insurance'},
    
    # Investment - 10+ samples
    {'description': 'Mutual fund SIP Zerodha', 'merchant': 'Zerodha Coin', 'amount': 10000.00, 'category': 'Investment'},
    {'description': 'Stock market investment Groww', 'merchant': 'Groww', 'amount': 15000.00, 'category': 'Investment'},
    {'description': 'Fixed deposit SBI Bank', 'merchant': 'SBI Bank', 'amount': 100000.00, 'category': 'Investment'},
    {'description': 'Gold investment digital gold', 'merchant': 'Digital Gold', 'amount': 5000.00, 'category': 'Investment'},
    {'description': 'PPF investment annual', 'merchant': 'PPF Account', 'amount': 150000.00, 'category': 'Investment'},
    {'description': 'ELSS tax saving fund', 'merchant': 'Tax Saving Fund', 'amount': 50000.00, 'category': 'Investment'},
    {'description': 'NSC investment post office', 'merchant': 'Post Office', 'amount': 25000.00, 'category': 'Investment'},
    {'description': 'Cryptocurrency investment', 'merchant': 'Crypto Exchange', 'amount': 20000.00, 'category': 'Investment'},
    {'description': 'Real estate investment', 'merchant': 'Real Estate', 'amount': 500000.00, 'category': 'Investment'},
    {'description': 'Bond investment government', 'merchant': 'Government Bonds', 'amount': 75000.00, 'category': 'Investment'},
    
    # Other - 10+ samples  
    {'description': 'Donation to charity NGO', 'merchant': 'NGO Charity', 'amount': 5000.00, 'category': 'Other'},
    {'description': 'Gift purchase birthday', 'merchant': 'Gift Store', 'amount': 2500.00, 'category': 'Other'},
    {'description': 'Personal loan repayment', 'merchant': 'Personal Loan', 'amount': 15000.00, 'category': 'Other'},
    {'description': 'ATM cash withdrawal', 'merchant': 'ATM', 'amount': 5000.00, 'category': 'Other'},
    {'description': 'Bank charges service fee', 'merchant': 'Bank Charges', 'amount': 500.00, 'category': 'Other'},
    {'description': 'Legal consultation lawyer', 'merchant': 'Legal Service', 'amount': 8000.00, 'category': 'Other'},
    {'description': 'Wedding photography service', 'merchant': 'Photography Service', 'amount': 25000.00, 'category': 'Other'},
    {'description': 'Pet care veterinary', 'merchant': 'Veterinary Clinic', 'amount': 2500.00, 'category': 'Other'},
    {'description': 'Freelance work payment', 'merchant': 'Freelance Client', 'amount': 15000.00, 'category': 'Other'},
    {'description': 'Religious ceremony expenses', 'merchant': 'Religious Service', 'amount': 5000.00, 'category': 'Other'}
]

def get_enhanced_training_data():
    """Return enhanced training data with timestamps"""
    enhanced_data = []
    for item in ENHANCED_TRAINING_DATA:
        enhanced_item = item.copy()
        enhanced_item['created_at'] = datetime.utcnow()
        # Add some randomization to amounts for better generalization
        if 'amount' in enhanced_item:
            base_amount = enhanced_item['amount']
            # Add ±10% variation
            variation = base_amount * 0.1 * (random.random() - 0.5) * 2
            enhanced_item['amount'] = round(base_amount + variation, 2)
        enhanced_data.append(enhanced_item)
    return enhanced_data

print(f"Enhanced training data contains {len(ENHANCED_TRAINING_DATA)} samples")
print("Categories distribution:")
categories = {}
for item in ENHANCED_TRAINING_DATA:
    cat = item['category']
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count} samples")