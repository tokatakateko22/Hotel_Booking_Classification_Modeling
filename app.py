from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (assuming it's a pickle file)
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [
        int(request.form['number_of_adults']),
        int(request.form['number_of_children']),
        int(request.form['number_of_weekend_nights']),
        int(request.form['number_of_week_nights']),
        request.form['type_of_meal'],
        request.form['room_type'],
        int(request.form['lead_time']),
        float(request.form['average_price']),
        int(request.form['special_requests']),
        request.form['date_of_reservation']
    ]

    # Convert categorical features to numerical if necessary (example code, may need modification)
    # type_of_meal_mapping = {'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3}
    # room_type_mapping = {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3}
    # features[4] = type_of_meal_mapping.get(features[4], 0)
    # features[5] = room_type_mapping.get(features[5], 0)

    # Ensure features are in the correct format
    features = np.array(features).reshape(1, -1)

    # Predict booking status
    prediction = model.predict(features)[0]
    prediction_text = "Cancelled" if prediction == 1 else "Not Cancelled"

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
