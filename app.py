from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Load the trained model
model = load_model('final_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the features from the incoming JSON data
    input_data = {
        'Location': [data['Location']],
        'Year': [data['Year']],
        'Kilometers_Driven': [data['Kilometers_Driven']],
        'Fuel_Type': [data['Fuel_Type']],
        'Transmission': [data['Transmission']],
        'Owner_Type': [data['Owner_Type']],
        'Mileage': [data['Mileage']],
        'Engine': [data['Engine']],
        'Power': [data['Power']],
        'Seats': [data['Seats']]
    }

    # Convert input data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Make prediction using the model
    prediction = predict_model(model, data=input_df)

    # Extract the predicted price from the 'prediction_label' column and convert to float
    predicted_price = float(prediction['prediction_label'][0])

    # Return the prediction as a JSON response
    return jsonify({'prediction': round(predicted_price, 2)})

if __name__ == '__main__':
    app.run(debug=True)
