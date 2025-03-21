import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import json
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_data(file_path):
    # Load data from file (assuming tab-separated format based on your data)
    df = pd.read_csv(file_path, sep='\t')
    
    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Create a year column to help with time-based predictions
    df['Year'] = df['Date'].dt.year - df['Date'].dt.year.min() + 1
    
    return df

# Step 2: Prepare features and target variables
def prepare_data(df):
    # Features will be: Week, Temperature, Humidity, LightLevel, Year
    # Only one vine model now
    
    # Prepare features
    features = ['Week', 'Temperature(C)', 'Humidity(%)', 'LightLevel(lux)', 'Year']
    
    # Dictionary to store model for the vine
    model = {}
    
    # Create model for vine length
    X = df[features]
    y_length = df['Vine_Length(m)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_length, test_size=0.2, random_state=42)
    
    # Train model for length
    model_length = RandomForestRegressor(n_estimators=100, random_state=42)
    model_length.fit(X_train, y_train)
    
    # Create model for leaves
    y_leaves = df['Vine_Leaves']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_leaves, test_size=0.2, random_state=42)
    
    # Train model for leaves
    model_leaves = RandomForestRegressor(n_estimators=100, random_state=42)
    model_leaves.fit(X_train, y_train)
    
    # Store models
    model['Length'] = model_length
    model['Leaves'] = model_leaves
    
    # Evaluate models
    print(f"Model for Vine Length:")
    y_pred = model_length.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    
    print(f"Model for Vine Leaves:")
    y_pred = model_leaves.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 50)
    
    return model

# Step 3: Visualize growth trends
def visualize_growth(df):
    plt.figure(figsize=(12, 8))
    
    # Plot length growth over time
    plt.subplot(2, 1, 1)
    plt.plot(df['Week'], df['Vine_Length(m)'], label='Vine Length', color='green')
    plt.xlabel('Week Number')
    plt.ylabel('Length (m)')
    plt.title('Vanilla Vine Length Growth Over Time')
    plt.legend()
    
    # Plot leaves growth over time
    plt.subplot(2, 1, 2)
    plt.plot(df['Week'], df['Vine_Leaves'], label='Vine Leaves', color='green')
    plt.xlabel('Week Number')
    plt.ylabel('Number of Leaves')
    plt.title('Vanilla Vine Leaves Growth Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/vanilla_growth_trends.png')
    plt.close()

# Step 4: Create a function to predict future growth
def predict_future_growth(model, target_week, temp=24.5, humidity=75.0, light=30000):
    # Calculate the year based on weeks (approximate)
    target_year = target_week // 52 + 1
    
    # Create a dataframe for the input features
    input_df = pd.DataFrame({
        'Week': [target_week],
        'Temperature(C)': [temp],
        'Humidity(%)': [humidity],
        'LightLevel(lux)': [light],
        'Year': [target_year]
    })
    
    # Make predictions
    length_model = model['Length']
    predicted_length = length_model.predict(input_df)[0]
    
    leaves_model = model['Leaves']
    predicted_leaves = leaves_model.predict(input_df)[0]
    
    # Store predictions
    predictions = {
        'Length': predicted_length,
        'Leaves': int(round(predicted_leaves))
    }
    
    return predictions

# Step 5: Save models for later use
def save_models(model, filename='static/vanilla_growth_model.pkl'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Step 6: Load saved models
def load_models(filename='static/vanilla_growth_model.pkl'):
    return joblib.load(filename)

# Flask routes to serve the web interface
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Load models
        model = load_models()
        
        # Make prediction
        target_week = int(data.get('targetWeek'))
        temp = float(data.get('temperature', 24.5))
        humidity = float(data.get('humidity', 75.0))
        light = float(data.get('lightLevel', 30000))
        
        predictions = predict_future_growth(model, target_week, temp, humidity, light)
        
        return jsonify({
            'success': True,
            'predictions': {
                'length': round(float(predictions['Length']), 2),
                'leaves': predictions['Leaves']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/train', methods=['POST'])
def train():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        file_path = 'uploads/data.txt'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        file.save(file_path)
        
        # Load and process data
        df = load_data(file_path)
        
        # Visualize the data
        visualize_growth(df)
        
        # Train models
        model = prepare_data(df)
        
        # Save models
        save_models(model)
        
        return jsonify({'success': True, 'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def init_app():
    # Ensure required directories exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Create the HTML, CSS, and JavaScript files for the web interface
    create_web_files()

def create_web_files():
    # HTML file
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vanilla Vine Growth Prediction</title>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>Vanilla Vine Growth Prediction System</h1>
        
        <div class="card">
            <h2>1. Train Model with Data</h2>
            <form id="trainForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="dataFile">Upload Data File (Tab-separated):</label>
                    <input type="file" id="dataFile" name="dataFile" accept=".txt,.csv">
                </div>
                <button type="submit" id="trainButton">Train Model</button>
            </form>
            <div id="trainResult" class="result-box"></div>
        </div>
        
        <div class="card">
            <h2>2. Predict Future Growth</h2>
            <form id="predictForm">
                <div class="form-group">
                    <label for="targetWeek">Target Week Number:</label>
                    <input type="number" id="targetWeek" name="targetWeek" min="1" required>
                </div>
                
                <h3>Environmental Factors (Optional)</h3>
                <div class="form-group">
                    <label for="temperature">Average Temperature (°C):</label>
                    <input type="number" id="temperature" name="temperature" step="0.1" placeholder="24.5">
                </div>
                <div class="form-group">
                    <label for="humidity">Average Humidity (%):</label>
                    <input type="number" id="humidity" name="humidity" step="0.1" placeholder="75.0">
                </div>
                <div class="form-group">
                    <label for="lightLevel">Average Light Level (lux):</label>
                    <input type="number" id="lightLevel" name="lightLevel" step="100" placeholder="30000">
                </div>
                
                <button type="submit" id="predictButton">Predict Growth</button>
            </form>
            <div id="predictResult" class="result-box"></div>
        </div>
        
        <div class="card">
            <h2>Growth Trends Visualization</h2>
            <div class="visualization">
                <img id="growthChart" src="" alt="Vanilla Growth Trends" style="display: none; max-width: 100%;">
                <p id="noVisualization">No visualization available. Train the model first.</p>
            </div>
        </div>
    </div>
    
    <script src="static/script.js"></script>
</body>
</html>"""

    # CSS file
    css_content = """* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
    padding: 20px;
}

.container {
    max-width: 900px;
    margin: 0 auto;
}

h1 {
    color: #2e7d32;
    text-align: center;
    margin-bottom: 30px;
}

h2 {
    color: #388e3c;
    margin-bottom: 15px;
}

h3 {
    color: #388e3c;
    margin: 15px 0 10px;
    font-size: 1rem;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 30px;
}

.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="file"],
input[type="number"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

button {
    background-color: #43a047;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2e7d32;
}

.result-box {
    margin-top: 15px;
    padding: 15px;
    border-radius: 4px;
    display: none;
}

.success {
    background-color: #e8f5e9;
    border: 1px solid #a5d6a7;
    color: #2e7d32;
    display: block;
}

.error {
    background-color: #ffebee;
    border: 1px solid #ef9a9a;
    color: #c62828;
    display: block;
}

.visualization {
    text-align: center;
    margin-top: 15px;
}

.visualization img {
    max-width: 100%;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}"""

    # JavaScript file
    js_content = """document.addEventListener('DOMContentLoaded', function() {
    // Check if we have a visualization
    checkVisualization();
    
    // Train form submission
    document.getElementById('trainForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('dataFile');
        const file = fileInput.files[0];
        
        if (!file) {
            showResult('trainResult', false, 'Please select a data file.');
            return;
        }
        
        // Disable button and show loading state
        const trainButton = document.getElementById('trainButton');
        trainButton.disabled = true;
        trainButton.textContent = 'Training...';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/train', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showResult('trainResult', true, 'Model trained successfully!');
                checkVisualization();
            } else {
                showResult('trainResult', false, `Error: ${data.error}`);
            }
        })
        .catch(error => {
            showResult('trainResult', false, `Error: ${error.message}`);
        })
        .finally(() => {
            // Re-enable button
            trainButton.disabled = false;
            trainButton.textContent = 'Train Model';
        });
    });
    
    // Predict form submission
    document.getElementById('predictForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const targetWeek = document.getElementById('targetWeek').value;
        const temperature = document.getElementById('temperature').value || '24.5';
        const humidity = document.getElementById('humidity').value || '75.0';
        const lightLevel = document.getElementById('lightLevel').value || '30000';
        
        // Disable button and show loading state
        const predictButton = document.getElementById('predictButton');
        predictButton.disabled = true;
        predictButton.textContent = 'Predicting...';
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                targetWeek: targetWeek,
                temperature: temperature,
                humidity: humidity,
                lightLevel: lightLevel
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const resultHTML = `
                    <h3>Prediction Results for Week ${targetWeek}:</h3>
                    <p><strong>Predicted Vine Length:</strong> ${data.predictions.length} meters</p>
                    <p><strong>Predicted Vine Leaves:</strong> ${data.predictions.leaves}</p>
                `;
                document.getElementById('predictResult').innerHTML = resultHTML;
                document.getElementById('predictResult').className = 'result-box success';
                document.getElementById('predictResult').style.display = 'block';
            } else {
                showResult('predictResult', false, `Error: ${data.error}`);
            }
        })
        .catch(error => {
            showResult('predictResult', false, `Error: ${error.message}`);
        })
        .finally(() => {
            // Re-enable button
            predictButton.disabled = false;
            predictButton.textContent = 'Predict Growth';
        });
    });
    
    // Function to show result message
    function showResult(elementId, isSuccess, message) {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = isSuccess ? 'result-box success' : 'result-box error';
        element.style.display = 'block';
    }
    
    // Function to check if visualization exists
    function checkVisualization() {
        const img = document.getElementById('growthChart');
        const noViz = document.getElementById('noVisualization');
        
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        fetch(`static/vanilla_growth_trends.png?t=${timestamp}`, { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    img.src = `static/vanilla_growth_trends.png?t=${timestamp}`;
                    img.style.display = 'block';
                    noViz.style.display = 'none';
                } else {
                    img.style.display = 'none';
                    noViz.style.display = 'block';
                }
            })
            .catch(() => {
                img.style.display = 'none';
                noViz.style.display = 'block';
            });
    }
});"""

    # Write files
    with open('static/index.html', 'w') as f:
        f.write(html_content)
    
    with open('static/style.css', 'w') as f:
        f.write(css_content)
    
    with open('static/script.js', 'w') as f:
        f.write(js_content)

# Main function to run the entire process
def main():
    print("Vanilla Vine Growth Prediction System")
    print("=====================================")
    
    # Initialize the app
    init_app()
    
    # Run the Flask app
    print("Starting web server...")
    print("Open http://127.0.0.1:5000 in your web browser")
    app.run(debug=True)

if __name__ == "__main__":
    main()