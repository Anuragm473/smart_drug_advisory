from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
from fuzzywuzzy import fuzz, process
import plotly.graph_objects as go
import plotly.utils
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values
    desc = " ".join(desc) if len(desc) > 0 else "No description available."

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    pre = pre[0].tolist() if len(pre) > 0 else []

    med = medications[medications['Disease'] == dis]['Medication'].values
    med = med.tolist() if len(med) > 0 else []

    die = diets[diets['Disease'] == dis]['Diet'].values
    die = die.tolist() if len(die) > 0 else []

    wrkout = workout[workout['disease'] == dis]['workout'].values
    wrkout = wrkout.tolist() if len(wrkout) > 0 else []

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def preprocess_symptom(symptom):
    """Preprocess symptom text for better matching."""
    # Convert to lowercase and remove extra spaces
    processed = symptom.lower().strip()
    # Replace underscores and hyphens with spaces
    processed = processed.replace('_', ' ').replace('-', ' ')
    # Remove any special characters
    processed = ''.join(c for c in processed if c.isalnum() or c.isspace())
    # Remove multiple spaces
    processed = ' '.join(processed.split())
    return processed

def get_best_symptom_match(input_symptom, symptoms_dict):
    """
    Get the best matching symptom using multiple fuzzy matching strategies.
    Returns tuple of (best_match, score, match_type)
    """
    input_symptom = preprocess_symptom(input_symptom)
    
    # Create a preprocessed version of the symptoms dictionary
    processed_symptoms = {preprocess_symptom(k): k for k in symptoms_dict.keys()}
    
    # Try exact match first (after preprocessing)
    if input_symptom in processed_symptoms:
        return processed_symptoms[input_symptom], 100, "exact"
    
    # Try different fuzzy matching strategies
    matches = []
    
    # Regular ratio with processed symptoms
    best_match, score = process.extractOne(input_symptom, processed_symptoms.keys(), scorer=fuzz.ratio)
    if best_match:
        matches.append((processed_symptoms[best_match], score, "ratio"))
    
    # Token sort ratio (better for words in different order)
    best_match_sort, score_sort = process.extractOne(input_symptom, processed_symptoms.keys(), scorer=fuzz.token_sort_ratio)
    if best_match_sort:
        matches.append((processed_symptoms[best_match_sort], score_sort, "token_sort"))
    
    # Partial ratio (better for substring matches)
    best_match_partial, score_partial = process.extractOne(input_symptom, processed_symptoms.keys(), scorer=fuzz.partial_ratio)
    if best_match_partial:
        matches.append((processed_symptoms[best_match_partial], score_partial, "partial"))
    
    if not matches:
        return None, 0, None
    
    # Get the best match across all strategies
    best_match, score, match_type = max(matches, key=lambda x: x[1])
    
    # Lower the threshold for common symptoms
    common_symptoms = {'weight gain', 'weight loss', 'fever', 'cough', 'headache', 'pain'}
    if input_symptom in common_symptoms:
        threshold = 70
    else:
        threshold = {
            "exact": 100,
            "ratio": 85,
            "token_sort": 85,
            "partial": 90
        }.get(match_type, 85)
    
    if score >= threshold:
        return best_match, score, match_type
    return None, score, match_type

def calculate_severity(matched_symptoms, predicted_disease):
    """
    Calculate disease severity based on:
    1. Number of symptoms present
    2. Critical symptom weights
    3. Disease-specific severity factors
    4. Symptom combinations
    Returns severity score (0-100) and contributing factors.
    """
    # Define critical symptoms that indicate high severity
    critical_symptoms = {
        'chest_pain': 0.9,
        'breathlessness': 0.9,
        'coma': 1.0,
        'altered_sensorium': 0.9,
        'high_fever': 0.8,
        'stomach_bleeding': 0.9,
        'acute_liver_failure': 1.0,
        'swelling_of_stomach': 0.8,
        'bloody_stool': 0.8,
        'yellowing_of_eyes': 0.8,
        'loss_of_consciousness': 1.0
    }

    # Define disease-specific severity weights
    disease_symptom_weights = {
        'Heart attack': {
            'chest_pain': 1.0, 
            'breathlessness': 0.9,
            'sweating': 0.7,
            'dizziness': 0.6,
            'fast_heart_rate': 0.8
        },
        'Pneumonia': {
            'breathlessness': 0.9,
            'high_fever': 0.8,
            'cough': 0.7,
            'chest_pain': 0.8,
            'phlegm': 0.6
        },
        'Diabetes': {
            'fatigue': 0.7,
            'weight_loss': 0.6,
            'restlessness': 0.5,
            'lethargy': 0.5,
            'irregular_sugar_level': 0.9,
            'excessive_hunger': 0.8,
            'polyuria': 0.7
        },
        'Tuberculosis': {
            'cough': 0.8,
            'blood_in_sputum': 0.9,
            'weight_loss': 0.7,
            'breathlessness': 0.8,
            'high_fever': 0.7
        },
        'Hepatitis': {
            'yellowish_skin': 0.8,
            'yellowing_of_eyes': 0.8,
            'dark_urine': 0.7,
            'acute_liver_failure': 1.0,
            'abdominal_pain': 0.6
        },
        'Malaria': {
            'high_fever': 0.8,
            'chills': 0.7,
            'sweating': 0.6,
            'headache': 0.6,
            'vomiting': 0.5
        }
    }

    # Calculate base severity score
    severity_score = 0
    max_possible_score = 0
    contributing_factors = []
    
    # Check for critical symptoms first
    critical_score = 0
    for symptom in matched_symptoms:
        if symptom in critical_symptoms:
            critical_score += critical_symptoms[symptom]
            contributing_factors.append({
                'symptom': symptom,
                'weight': critical_symptoms[symptom],
                'contribution': critical_symptoms[symptom] * 100,
                'type': 'Critical Symptom'
            })

    # Get disease-specific weights
    weights = disease_symptom_weights.get(predicted_disease, {})
    
    # Calculate disease-specific severity
    disease_score = 0
    for symptom in matched_symptoms:
        weight = weights.get(symptom, 0.5)  # Default weight if symptom not specified
        disease_score += weight
        contributing_factors.append({
            'symptom': symptom,
            'weight': weight,
            'contribution': (weight / len(matched_symptoms)) * 100,
            'type': 'Disease Specific'
        })

    # Consider number of symptoms (more symptoms = higher severity)
    symptom_count_factor = min(len(matched_symptoms) / 5, 1.0)  # Cap at 5 symptoms
    
    # Calculate final severity score (0-100)
    # Weight distribution: 40% critical symptoms, 40% disease-specific, 20% symptom count
    final_score = (
        (critical_score * 40) +
        (disease_score / len(matched_symptoms) * 40) +
        (symptom_count_factor * 20)
    )
    
    # Ensure score is between 0 and 100
    final_score = min(max(final_score, 0), 100)
    
    return final_score, contributing_factors

def create_severity_gauge(severity_score):
    """Create a Plotly gauge chart for disease severity."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = severity_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Disease Severity", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "white",
        font = {'color': "darkblue", 'family': "Arial"}
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    matched_symptoms = []
    unmatched_symptoms = []
    
    for symptom in patient_symptoms:
        if not symptom.strip():  # Skip empty symptoms
            continue
            
        best_match, score, match_type = get_best_symptom_match(symptom, symptoms_dict)
        
        if best_match is not None:
            try:
                input_vector[symptoms_dict[best_match]] = 1
                matched_symptoms.append(best_match)
                print(f"Matched '{symptom}' to '{best_match}' with {score}% confidence using {match_type} matching")
            except KeyError as e:
                print(f"Warning: Matched symptom '{best_match}' not found in symptoms dictionary")
                unmatched_symptoms.append(symptom)
        else:
            print(f"Warning: Could not find a good match for symptom '{symptom}' (best score: {score}%)")
            unmatched_symptoms.append(symptom)
    
    if not matched_symptoms:
        raise ValueError("No valid symptoms could be matched. Please check your input.")
    
    predicted_disease = diseases_list.get(svc.predict([input_vector])[0], "Unknown Disease")
    severity_score, contributing_factors = calculate_severity(matched_symptoms, predicted_disease)
    severity_graph = create_severity_gauge(severity_score)
    
    return predicted_disease, severity_score, severity_graph, contributing_factors

def get_severity_suggestions(severity_score, predicted_disease):
    """Get detailed suggestions based on disease severity level"""
    emergency_diseases = {'Heart attack', 'Pneumonia', 'Acute liver failure'}
    
    suggestions = {
        'low': [
            "Monitor your symptoms and maintain a symptom diary",
            "Rest adequately and stay hydrated",
            "Follow basic preventive measures",
            "Consider over-the-counter medications as appropriate",
            "Schedule a routine check-up if symptoms persist"
        ],
        'medium': [
            "Schedule an appointment with your healthcare provider within 24-48 hours",
            "Monitor vital signs if possible (temperature, blood pressure)",
            "Follow prescribed medications strictly",
            "Avoid strenuous activities and get adequate rest",
            "Have someone check on you periodically"
        ],
        'high': [
            "⚠️ SEEK IMMEDIATE MEDICAL ATTENTION ⚠️",
            "Contact emergency services or visit nearest emergency room",
            "Do not drive yourself - ask someone to accompany you",
            "Have your medical history and medication list ready",
            "Continue any critical medications unless told otherwise by a doctor"
        ]
    }
    
    # Determine severity level
    if predicted_disease in emergency_diseases:
        severity_level = 'high'  # Override for emergency conditions
    elif severity_score < 30:
        severity_level = 'low'
    elif severity_score < 70:
        severity_level = 'medium'
    else:
        severity_level = 'high'
        
    return severity_level, suggestions[severity_level]

# Web interface route
@app.route('/')
def index():
    return render_template('index.html')

# API route for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    
    # Process symptoms
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
    
    predicted_disease, severity_score, severity_graph, contributing_factors = get_predicted_value(user_symptoms)
    dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
    
    # Get severity-based suggestions
    severity_level, severity_suggestions = get_severity_suggestions(severity_score, predicted_disease)
    
    return jsonify({
        "disease": predicted_disease,
        "description": dis_des,
        "precautions": precautions,
        "medications": medications,
        "diets": rec_diet,
        "suggestions": workout,
        "severity_score": severity_score,
        "severity_graph": severity_graph,
        "contributing_factors": contributing_factors,
        "severity_level": severity_level,
        "severity_suggestions": severity_suggestions
    })

# Web form route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form.get('symptoms')
        
        if not symptoms:
            return render_template('index.html', message="No symptoms provided. Please enter valid symptoms.")
        
        # Process symptoms
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        
        predicted_disease, severity_score, severity_graph, contributing_factors = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
        # Get severity-based suggestions
        severity_level, severity_suggestions = get_severity_suggestions(severity_score, predicted_disease)
        
        return render_template('index.html', 
                             predicted_disease=predicted_disease,
                             dis_des=dis_des,
                             my_precautions=precautions,
                             medications=medications,
                             my_diet=rec_diet,
                             workout=workout,
                             severity_score=severity_score,
                             severity_graph=severity_graph,
                             contributing_factors=contributing_factors,
                             severity_level=severity_level,
                             severity_suggestions=severity_suggestions,
                             show_results=True)
                             
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return render_template('index.html', 
                             message=f"An error occurred while processing your symptoms. Please try again. Error: {str(e)}",
                             show_results=False)

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 