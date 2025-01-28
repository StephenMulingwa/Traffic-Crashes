from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Load the trained model
with open('../Traffic_Car_Prediction.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Define the preprocessing steps (same as during training)
numerical_features = ['POSTED_SPEED_LIMIT', 'NUM_UNITS', 'INJURIES_TOTAL', 'INJURIES_FATAL', 'CRASH_HOUR', 'CRASH_MONTH', 'LATITUDE', 'LONGITUDE']
categorical_features = ['TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT', 'REPORT_TYPE', 'CRASH_TYPE', 'DAMAGE', 'STREET_DIRECTION', 'MOST_SEVERE_INJURY', 'CRASH_DAY_OF_WEEK']

numerical_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())  # Standardize numerical features
])

categorical_pipeline = Pipeline(steps=[ 
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Encode categorical features
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        input_data = {
            'POSTED_SPEED_LIMIT': int(request.form['POSTED_SPEED_LIMIT']),
            'TRAFFIC_CONTROL_DEVICE': request.form['TRAFFIC_CONTROL_DEVICE'],
            'DEVICE_CONDITION': request.form['DEVICE_CONDITION'],
            'WEATHER_CONDITION': request.form['WEATHER_CONDITION'],
            'LIGHTING_CONDITION': request.form['LIGHTING_CONDITION'],
            'FIRST_CRASH_TYPE': request.form['FIRST_CRASH_TYPE'],
            'TRAFFICWAY_TYPE': request.form['TRAFFICWAY_TYPE'],
            'ALIGNMENT': request.form['ALIGNMENT'],
            'ROADWAY_SURFACE_COND': request.form['ROADWAY_SURFACE_COND'],
            'ROAD_DEFECT': request.form['ROAD_DEFECT'],
            'REPORT_TYPE': request.form['REPORT_TYPE'],
            'CRASH_TYPE': request.form['CRASH_TYPE'],
            'DAMAGE': request.form['DAMAGE'],
            'STREET_DIRECTION': request.form['STREET_DIRECTION'],
            'NUM_UNITS': int(request.form['NUM_UNITS']),
            'MOST_SEVERE_INJURY': request.form['MOST_SEVERE_INJURY'],
            'INJURIES_TOTAL': float(request.form['INJURIES_TOTAL']),
            'INJURIES_FATAL': float(request.form['INJURIES_FATAL']),
            'CRASH_HOUR': int(request.form['CRASH_HOUR']),
            'CRASH_DAY_OF_WEEK': request.form['CRASH_DAY_OF_WEEK'],
            'CRASH_MONTH': int(request.form['CRASH_MONTH']),
            'LATITUDE': float(request.form['LATITUDE']),
            'LONGITUDE': float(request.form['LONGITUDE']),
        }

        # Convert input data to a pandas DataFrame
        features = pd.DataFrame([input_data])

        # Ensure preprocessor is fitted before transformation (only fit once at start)
        if not hasattr(preprocessor, 'transformers_'):
            # Fit the preprocessor on a sample dataset
            sample_data = pd.DataFrame([{
                'POSTED_SPEED_LIMIT': 30, 'TRAFFIC_CONTROL_DEVICE': 'TRAFFIC SIGNAL', 'DEVICE_CONDITION': 'FUNCTIONING PROPERLY', 'WEATHER_CONDITION': 'CLEAR',
                'LIGHTING_CONDITION': 'DUSK', 'FIRST_CRASH_TYPE': 'ANGLE', 'TRAFFICWAY_TYPE': 'FIVE POINT, OR MORE', 'ALIGNMENT': 'STRAIGHT AND LEVEL',
                'ROADWAY_SURFACE_COND': 'DRY', 'ROAD_DEFECT': 'NO DEFECTS', 'REPORT_TYPE': 'ON SCENE', 'CRASH_TYPE': 'INJURY AND / OR TOW DUE TO CRASH',
                'DAMAGE': 'OVER $1,500', 'STREET_DIRECTION': 'S', 'NUM_UNITS': 2, 'MOST_SEVERE_INJURY': 'INCAPACITATING INJURY', 'INJURIES_TOTAL': 3.0,
                'INJURIES_FATAL': 0.0, 'CRASH_HOUR': 19, 'CRASH_DAY_OF_WEEK': '3', 'CRASH_MONTH': 9, 'LATITUDE': 41.85412, 'LONGITUDE': -87.665902
            }])
            preprocessor.fit(sample_data)  # Fit on a sample dataset

        # Preprocess the input data
        features_preprocessed = preprocessor.transform(features)  # Use transform, not fit_transform

        # Pad the features array to match the expected shape
        padded_features = np.zeros((1, 175))  # Create an array of zeros with shape (1, 175)
        padded_features[:, :features_preprocessed.shape[1]] = features_preprocessed  # Fill the initial part with the preprocessed features
        
        # Reshape to add an additional layer (e.g., (1, 1, 175))
        reshaped_features = padded_features.reshape(1, 1, 175)  # Add a layer (e.g., a time-step or sequence length)

        # Make prediction
        prediction = model.predict(reshaped_features)  # No need to reshape

        # Decode the prediction (if necessary)
        predicted_class = np.argmax(prediction, axis=1)
        print(f'pred: {predicted_class}')

        # Return the prediction result
        # List of class names
        class_names = [
            'FAILING TO REDUCE SPEED TO AVOID CRASH', 'UNABLE TO DETERMINE',
            'IMPROPER BACKING', 'IMPROPER LANE USAGE',
            'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE',
            'FAILING TO YIELD RIGHT-OF-WAY', 'IMPROPER OVERTAKING/PASSING',
            'DISREGARDING TRAFFIC SIGNALS', 'WEATHER',
            'IMPROPER TURNING/NO SIGNAL', 'DRIVING ON WRONG SIDE/WRONG WAY',
            'NOT APPLICABLE', 'FOLLOWING TOO CLOSELY',
            'DISTRACTION - FROM INSIDE VEHICLE', 'ANIMAL',
            'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)', 'TEXTING',
            'EQUIPMENT - VEHICLE CONDITION',
            'DISREGARDING OTHER TRAFFIC SIGNS', 'DISREGARDING STOP SIGN',
            'OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER',
            'ROAD CONSTRUCTION/MAINTENANCE',
            'DISTRACTION - FROM OUTSIDE VEHICLE',
            'EXCEEDING SAFE SPEED FOR CONDITIONS',
            'PHYSICAL CONDITION OF DRIVER', 'DISREGARDING YIELD SIGN',
            'ROAD ENGINEERING/SURFACE/MARKING DEFECTS', 'TURNING RIGHT ON RED',
            'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)',
            'EXCEEDING AUTHORIZED SPEED LIMIT', 'DISREGARDING ROAD MARKINGS',
            'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST',
            'CELL PHONE USE OTHER THAN TEXTING',
            'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)',
            'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)',
            'OBSTRUCTED CROSSWALKS', 'RELATED TO BUS STOP',
            'BICYCLE ADVANCING LEGALLY ON RED LIGHT',
            'PASSING STOPPED SCHOOL BUS',
            'MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT'
        ]

        # Decode the prediction (index to class name)
        predicted_class_name = class_names[predicted_class[0]]

        # Return the prediction result with the class name
        return render_template('result.html', prediction=f"{predicted_class_name}")

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
