from flask import Flask, request, render_template
import pickle
import pandas as pd
from car_data_prep import prepare_data
import os

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    print("Form Data Received:", form_data)  # Debug print
    
    # Convert form data to DataFrame
    input_data = pd.DataFrame([form_data])
    print("Input DataFrame:", input_data)  # Debug print

    # Rename form columns to match expected DataFrame column names
    column_mapping = {
        'manufactor': 'manufactor',
        'model': 'model',
        'Year': 'Year',
        'Hand': 'Hand',
        'Gear': 'Gear',
        'capacity_Engine': 'capacity_Engine',
        'Engine_type': 'Engine_type',
        'Prev_ownership': 'Prev_ownership',
        'Curr_ownership': 'Curr_ownership',
        'Area': 'Area',
        'Km': 'Km'
    }
    input_data.rename(columns=column_mapping, inplace=True)
    print("Renamed DataFrame:", input_data)  # Debug print

    # Prepare the data
    prepared_data = prepare_data(input_data)
    print("Prepared DataFrame:", prepared_data)  # Debug print

    # Ensure the columns are in the correct order
    prepared_data = prepared_data.reindex(columns=model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2])
    print("Reindexed Prepared DataFrame:", prepared_data)  # Debug print

    # Export prepared_data as CSV for debugging
    #csv_path = os.path.join('debug', 'prepared_data.csv')
    #os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    #prepared_data.to_csv(csv_path, index=False)
    #print(f"Prepared data saved to {csv_path}")

    # Make prediction
    prediction = model.predict(prepared_data)
    
    # Get the predicted price
    predicted_price = prediction[0]
    
    return render_template('index.html', prediction_text=f'Predicted Vehicle Price: {predicted_price}')

if __name__ == '__main__':
    app.run(debug=True)
