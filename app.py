from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import mysql.connector
from datetime import datetime

try:
    with open('svc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    print("All necessary model components loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: One or more .pkl files not found. Run train_model.py first. Missing file: {e.filename}")
    exit()

app = Flask(__name__)

DB_CONFIG = {
    'user': 'root',           
    'password': '',           
    'host': 'localhost',
    'database': 'dry_eye_db',
    'port': '3306'           
}

def connect_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def preprocess_data(input_data_dict):
    """
    ইনপুট ডেটা ডিকশনারি নেয় এবং সেটিকে মডেল প্রেডিকশনের জন্য প্রস্তুত করে।
    - Gender এনকোডিং।
    - Y/N বাইনারি কলাম 1/0 তে রূপান্তর।
    - সংখ্যাগত কলাম স্কেলিং।
    - কলামের ক্রম মডেল ট্রেইনিংয়ের সাথে মিলিয়ে সাজানো।
    """
    data = pd.DataFrame([input_data_dict])
    
    
    if 'Gender' in data.columns:
        data['Gender'] = le_gender.transform(data['Gender'])
        
    binary_map = {'Y': 1, 'N': 0}
    binary_cols = [
        'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 
        'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 
        'Ongoing medication', 'Smart device before bed', 'Blue-light filter', 
        'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye'
    ]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.upper().map(binary_map)
            data[col].fillna(0, inplace=True) 
            
    numeric_cols = [
        'Age', 'Sleep duration', 'Sleep quality', 'Stress level', 
        'Systolic_BP', 'Diastolic_BP', 'Heart rate', 'Daily steps', 
        'Physical activity', 'Height', 'Weight', 'Average screen time'
    ]
    for col in numeric_cols:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                 print(f"Error converting {col} to numeric: {e}")
                 
    X_processed = data[model_columns].copy()

    if X_processed.isnull().any().any():
        X_processed.fillna(X_processed.mean(), inplace=True) 
        
   
    numerical_cols_to_scale = [col for col in model_columns if col in scaler.feature_names_in_]
    
    X_processed[numerical_cols_to_scale] = X_processed[numerical_cols_to_scale].astype(float)
    
    X_processed[numerical_cols_to_scale] = scaler.transform(X_processed[numerical_cols_to_scale])
    
    return X_processed.values

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<page_name>.html')
def serve_page(page_name):
    return render_template(f'{page_name}.html')

@app.route('/test_db')
def test_db():
    try:
        conn = connect_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            return jsonify({'message': 'Database connection successful!', 'result': result})
        else:
            return jsonify({'error': 'Failed to connect to database'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)
        
        #Form data  
        form_data = {
            # Personal Info
            'Gender': data.get('gender', 'F'), # F/M
            'Age': data.get('age', 23),#24
            'Height': data.get('height', 168),#161
            'Weight': data.get('weight',79 ),#69
            
            # Health Metrics
            'Sleep duration': data.get('sleep_duration', 9.3),
            'Sleep quality': data.get('sleep_quality', 3), # 1-5
            'Stress level': data.get('stress_level', 2), # 1-5
            'Systolic_BP': data.get('systolic_bp', 110),
            'Diastolic_BP': data.get('diastolic_bp', 66),
            'Heart rate': data.get('heart_rate', 89),
            'Daily steps': data.get('daily_steps', 5000),
            'Physical activity': data.get('physical_activity', 153),
            'Sleep disorder': data.get('sleep_disorder', 'N'), # Y/N
            'Wake up during night': data.get('wake_up_during_night', 'Y'), # Y/N
            'Feel sleepy during day': data.get('feel_sleepy_during_day', 'Y'), # Y/N
            
            # Lifestyle
            'Caffeine consumption': data.get('caffeine_consumption', 'N'), # Y/N
            'Alcohol consumption': data.get('alcohol_consumption', 'Y'), # Y/N
            'Smoking': data.get('smoking', 'N'), # Y/N
            'Medical issue': data.get('medical_issue', 'Y'), # Y/N
            'Ongoing medication': data.get('ongoing_medication', 'Y'), # Y/N
            'Smart device before bed': data.get('smart_device_before_bed', 'N'), # Y/N
            'Average screen time': data.get('average_screen_time', 3.1),
            'Blue-light filter': data.get('blue_light_filter', 'N'), # Y/N
            
            # Eye Symptoms
            'Discomfort Eye-strain': data.get('discomfort_eye_strain', 'N'), # Y/N
            'Redness in eye': data.get('redness_in_eye', 'Y'), # Y/N
            'Itchiness/Irritation in eye': data.get('itchiness_irritation_in_eye', 'N'), # Y/N
        }
        
        
        X_test_processed = preprocess_data(form_data)
        
     
        prediction_proba = model.predict_proba(X_test_processed)[0]
        
        
        threshold = 0.4
        
        dry_eye_probability = prediction_proba[1]
        
        print(f"প্রেডিকশন সম্ভাবনা: না = {prediction_proba[0]:.3f}, হ্যাঁ = {prediction_proba[1]:.3f}")
        print(f"ব্যবহৃত থ্রেশহোল্ড: {threshold}")
        
       
        if dry_eye_probability >= threshold:
            prediction_result = 'Y'
            confidence = dry_eye_probability * 100
        else:
            prediction_result = 'N'
            confidence = (1 - dry_eye_probability) * 100
        
      
        if prediction_result == 'Y':
            result_message = f"Dry Eye Disease (Confidence: {confidence:.2f}%)."
        else:
            result_message = f"No Dry Eye Disease. (Confidence: {confidence:.2f}%)."
            
        
        db_success = save_to_db(form_data, prediction_result)
        if not db_success:
            print("Warning: Data could not be saved to database")
        
        return jsonify({
            'prediction': prediction_result, 
            'message': result_message,
            'probability_yes': f"{dry_eye_probability:.3f}",
            'probability_no': f"{prediction_proba[0]:.3f}",
            'threshold_used': threshold
        })

    except ValueError as ve:
         print(f"Input Data Error: {ve}")
         return jsonify({'prediction': 'Error', 'message': f"Prediction failed due to invalid input data: {ve}"}), 400
    except Exception as e:
        print(f"An internal error occurred: {e}")
        return jsonify({'prediction': 'Error', 'message': f"Prediction failed due to an internal server error. Please check logs."}), 500

def save_to_db(data, prediction_result):
  
    conn = None
    try:
        conn = connect_db()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        
        columns = (
            'gender', 'age', 'height', 'weight', 'sleep_duration', 'sleep_quality', 
            'stress_level', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'daily_steps', 
            'physical_activity', 'sleep_disorder', 'wake_up_during_night', 
            'feel_sleepy_during_day', 'caffeine_consumption', 'alcohol_consumption', 
            'smoking', 'medical_issue', 'ongoing_medication', 'smart_device_before_bed', 
            'average_screen_time', 'blue_light_filter', 'discomfort_eye_strain', 
            'redness_in_eye', 'itchiness_irritation_in_eye', 'prediction_result'
        )
        
        
        values = (
            data.get('Gender', 'F'), data.get('Age', 0), data.get('Height', 0), 
            data.get('Weight', 0), data.get('Sleep duration', 0), 
            data.get('Sleep quality', 0), data.get('Stress level', 0), 
            data.get('Systolic_BP', 0), data.get('Diastolic_BP', 0), 
            data.get('Heart rate', 0), data.get('Daily steps', 0), 
            data.get('Physical activity', 0), data.get('Sleep disorder', 'N'), 
            data.get('Wake up during night', 'N'), data.get('Feel sleepy during day', 'N'), 
            data.get('Caffeine consumption', 'N'), data.get('Alcohol consumption', 'N'), 
            data.get('Smoking', 'N'), data.get('Medical issue', 'N'), 
            data.get('Ongoing medication', 'N'), data.get('Smart device before bed', 'N'), 
            data.get('Average screen time', 0), data.get('Blue-light filter', 'N'), 
            data.get('Discomfort Eye-strain', 'N'), data.get('Redness in eye', 'N'), 
            data.get('Itchiness/Irritation in eye', 'N'), prediction_result
        )

        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join(columns)
        
        query = f"INSERT INTO user_submissions ({column_names}) VALUES ({placeholders})"
        
        cursor.execute(query, values)
        conn.commit()
        print("Data successfully saved to database.")
        return True

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route('/create_table')
def create_table():
    try:
        conn = connect_db()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_submissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            gender VARCHAR(10),
            age INT,
            height FLOAT,
            weight FLOAT,
            sleep_duration FLOAT,
            sleep_quality INT,
            stress_level INT,
            systolic_bp INT,
            diastolic_bp INT,
            heart_rate INT,
            daily_steps INT,
            physical_activity INT,
            sleep_disorder ENUM('Y', 'N'),
            wake_up_during_night ENUM('Y', 'N'),
            feel_sleepy_during_day ENUM('Y', 'N'),
            caffeine_consumption ENUM('Y', 'N'),
            alcohol_consumption ENUM('Y', 'N'),
            smoking ENUM('Y', 'N'),
            medical_issue ENUM('Y', 'N'),
            ongoing_medication ENUM('Y', 'N'),
            smart_device_before_bed ENUM('Y', 'N'),
            average_screen_time FLOAT,
            blue_light_filter ENUM('Y', 'N'),
            discomfort_eye_strain ENUM('Y', 'N'),
            redness_in_eye ENUM('Y', 'N'),
            itchiness_irritation_in_eye ENUM('Y', 'N'),
            prediction_result ENUM('Y', 'N'),
            submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Table created successfully or already exists'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    
    app.run(host='127.0.0.1', port=5000, debug=True)