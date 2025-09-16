from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)

try:
    with open('logistic_regression_model.pkl', 'rb') as f_lr:
        lr_model = pickle.load(f_lr)
    with open('random_forest_model.pkl', 'rb') as f_rf:
        rf_model = pickle.load(f_rf)
    print("Pickle models loaded successfully")
except FileNotFoundError:
    print("Error: The .pkl model files were not found. Please run the training script to create them.")
    lr_model = None
    rf_model = None



@app.route('/')
def home():
    return render_template('html.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        user_input = json_data['data'][0]
        algorithm = json_data.get('algorithm', 'random_forest')


        default_row = {
            'Booking_ID': 'None', 'number of adults': 0, 'number of children': 0,
            'number of weekend nights': 0, 'number of week nights': 0,
            'type of meal': 'None', 'car parking space': 0, 'room type': 'None',
            'lead time': 0, 'market segment type': 'None', 'repeated': 0,
            'P-C': 0, 'P-not-C': 0, 'average price': 0.0, 'special requests': 0,
            'date of reservation': 'None'
        }


        default_row.update(user_input)

        full_df = pd.DataFrame([default_row])


        training_columns = [
            'Booking_ID', 'number of adults', 'number of children',
            'number of weekend nights', 'number of week nights', 'type of meal',
            'car parking space', 'room type', 'lead time', 'market segment type',
            'repeated', 'P-C', 'P-not-C', 'average price', 'special requests',
            'date of reservation'
        ]

        numeric_features = [
            'number of adults', 'number of children', 'number of weekend nights',
            'number of week nights', 'car parking space', 'lead time', 'repeated',
            'P-C', 'P-not-C', 'average price', 'special requests'
        ]

        for col in training_columns:
            if col in numeric_features:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
            else:
                full_df[col] = full_df[col].astype(str)

        full_df = full_df[training_columns]


        if algorithm == 'logistic_regression':
            model = lr_model
        else:
            model = rf_model

        # إجراء التوقع
        prediction = model.predict(full_df)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred in the /predict route: {e}")
        print(f"Data received from the browser: {request.get_json()}")
        print("--- END ERROR ---\n")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)