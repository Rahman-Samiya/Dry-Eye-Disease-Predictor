# Dry Eye Disease Prediction System

A machine learning web application that predicts whether a user has Dry Eye Disease based on their health and lifestyle parameters.

## Overview

This project uses a Support Vector Classifier (SVC) model trained on health data to predict Dry Eye Disease. The web interface allows users to input their health metrics and receive predictions with confidence scores.

## Features

- **Web-based prediction interface** - User-friendly form for inputting health data
- **Real-time predictions** - Get instant predictions with probability scores
- **Database storage** - Save predictions to MySQL database
- **REST API** - JSON-based API for predictions

## Project Structure

```
ML project/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── eye.py                 # Original Jupyter notebook analysis
├── Dry_Eye_Dataset.csv    # Training dataset
├── README.md              # This file
├── .qodo/                 # Internal files
├── templates/
│   ├── index.html         # Main web page
│   ├── styles.css         # Styling
│   └── script.js          # Client-side JavaScript
├── svc_model.pkl          # Trained SVC model
├── scaler.pkl             # StandardScaler for feature scaling
├── le_gender.pkl          # LabelEncoder for Gender
└── model_columns.pkl      # Model feature columns
```

## Requirements

- Python 3.8+
- Flask
- pandas
- numpy
- scikit-learn
- mysql-connector-python
- imbalanced-learn (for SMOTE)

## Installation

1. **Install dependencies:**
   ```bash
   pip install flask pandas numpy scikit-learn mysql-connector-python imbalanced-learn
   ```

2. **Set up MySQL database:**
   ```bash
   mysql -u root -p
   CREATE DATABASE dry_eye_db;
   ```

3. **Train the model:**
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   Open http://127.0.0.1:5000 in your browser

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web page |
| `/predict` | POST | Single prediction (JSON) |
| `/predict_csv` | POST | Batch prediction (CSV upload) |
| `/test_db` | GET | Test database connection |
| `/create_table` | GET | Create database table |

## Input Parameters

The prediction model uses the following input parameters:

### Personal Info
- **Gender**: Male (M) or Female (F)
- **Age**: 0-120 years
- **Height**: cm
- **Weight**: kg

### Health Metrics
- **Sleep duration**: hours (0-24)
- **Sleep quality**: 1-5 scale
- **Stress level**: 1-10 scale
- **Blood pressure**: Systolic/Diastolic (e.g., 120/80)
- **Heart rate**: bpm (30-200)
- **Daily steps**: count
- **Physical activity**: minutes/day

### Lifestyle
- **Sleep disorder**: Y/N
- **Wake up during night**: Y/N
- **Feel sleepy during day**: Y/N
- **Caffeine consumption**: Y/N
- **Alcohol consumption**: Y/N
- **Smoking**: Y/N
- **Medical issue**: Y/N
- **Ongoing medication**: Y/N
- **Smart device before bed**: Y/N
- **Average screen time**: hours/day
- **Blue-light filter**: Y/N

### Eye Symptoms
- **Discomfort Eye-strain**: Y/N
- **Redness in eye**: Y/N
- **Itchiness/Irritation in eye**: Y/N

## Output

The prediction returns:
- `prediction`: Y (Dry Eye Disease) or N (No Dry Eye Disease)
- `message`: Human-readable result message
- `probability_yes`: Probability of Dry Eye Disease (0-1)
- `probability_no`: Probability of No Dry Eye Disease (0-1)
- `threshold_used`: Classification threshold (default: 0.4)

## Example Usage

### Web Form
1. Fill in the form at http://127.0.0.1:5000
2. Click "Check" button
3. View prediction result

### API Call
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "F",
    "Age": 24,
    "Sleep duration": 9.5,
    "Sleep quality": 2,
    "Stress level": 1,
    "Blood pressure": "137/89",
    "Heart rate": 67,
    "Daily steps": 3000,
    "Physical activity": 31,
    "Height": 161,
    "Weight": 69,
    "Sleep disorder": "Y",
    "Wake up during night": "N",
    "Feel sleepy during day": "N",
    "Caffeine consumption": "N",
    "Alcohol consumption": "N",
    "Smoking": "N",
    "Medical issue": "Y",
    "Ongoing medication": "Y",
    "Smart device before bed": "N",
    "Average screen time": 8.7,
    "Blue-light filter": "N",
    "Discomfort Eye-strain": "Y",
    "Redness in eye": "Y",
    "Itchiness/Irritation in eye": "N"
  }'
```

## Model Details

- **Algorithm**: Support Vector Classifier (SVC)
- **Kernel**: Linear
- **Class balancing**: Enabled (class_weight='balanced')
- **Training**: Uses SMOTE for handling imbalanced data
- **Feature scaling**: StandardScaler

## Troubleshooting

### Database Connection Error
- Ensure MySQL is running
- Check database credentials in `app.py`
- Run `/create_table` endpoint to create required table

### Model Loading Error
- Run `python train_model.py` first to generate pickle files
- Ensure all .pkl files are in the project directory

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset source: Dry Eye Disease Dataset
- Model adapted from Jupyter notebook analysis
