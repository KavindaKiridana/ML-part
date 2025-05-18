import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

# Load models (same as original)
def get_models(data_file='vanilla_data.txt'):
    model_file = 'vanilla_growth_models.pkl'
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print("Training models for the first time. This may take a moment...")
        df = pd.read_csv(data_file, sep='\t')
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Year'] = df['Date'].dt.year - df['Date'].dt.year.min() + 1
        
        vine_names = ['Vine1', 'Vine2', 'Vine3', 'Vine4']
        features = ['Week', 'Temperature(C)', 'Humidity(%)', 'LightLevel(lux)', 'Year']
        
        models = {}
        
        for vine in vine_names:
            X = df[features]
            y_length = df[f'{vine}_Length(m)']
            model_length = RandomForestRegressor(n_estimators=100, random_state=42)
            model_length.fit(X, y_length)
            
            y_leaves = df[f'{vine}_Leaves']
            model_leaves = RandomForestRegressor(n_estimators=100, random_state=42)
            model_leaves.fit(X, y_leaves)
            
            models[f'{vine}_Length'] = model_length
            models[f'{vine}_Leaves'] = model_leaves
        
        joblib.dump(models, model_file)
        return models

# Prediction function (same as original)
def predict_growth(initial_week, target_week, temp, humidity, light, models):
    initial_year = initial_week // 52 + 1
    target_year = target_week // 52 + 1
    
    input_features = pd.DataFrame({
        'Week': [target_week],
        'Temperature(C)': [temp],
        'Humidity(%)': [humidity],
        'LightLevel(lux)': [light],
        'Year': [target_year]
    })
    
    predictions = {}
    for vine in ['Vine1', 'Vine2', 'Vine3', 'Vine4']:
        length_model = models[f'{vine}_Length']
        leaves_model = models[f'{vine}_Leaves']
        
        predicted_length = length_model.predict(input_features)[0]
        predicted_leaves = leaves_model.predict(input_features)[0]
        
        predictions[f'{vine}_Length'] = predicted_length
        predictions[f'{vine}_Leaves'] = int(round(predicted_leaves))
    
    return predictions

# Main program
def main():
    print("\nVanilla Vine Growth Predictor (Terminal Version)")
    print("Enter your current measurements to see future predictions\n")
    
    # Get user inputs
    current_week = int(input("Current Week Number (1-156): "))
    current_length = float(input("Current Vine Length (meters): "))
    current_leaves = int(input("Current Number of Leaves: "))
    target_week = int(input(f"Predict for Week Number ({current_week+1}-156): "))
    
    print("\nEnter Environmental Factors:")
    temperature = float(input("Average Temperature (°C, 20-30): "))
    humidity = float(input("Average Humidity (%, 60-90): "))
    light_level = int(input("Average Light Level (lux, 25000-35000): "))
    
    # Load models
    try:
        models = get_models()
        print("\nModels loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Make predictions
    predictions = predict_growth(current_week, target_week, temperature, humidity, light_level, models)
    
    # Display results
    print("\n=== Predicted Growth Results ===")
    print(f"Projection from Week {current_week} to Week {target_week}")
    
    print("\nVine\tLength (m)\tLeaves")
    print("--------------------------------")
    for i, vine in enumerate(['Vine1', 'Vine2', 'Vine3', 'Vine4']):
        print(f"Vine {i+1}\t{predictions[f'{vine}_Length']:.2f}\t\t{predictions[f'{vine}_Leaves']}")
    
    # Calculate growth
    length_growth = predictions['Vine1_Length'] - current_length
    leaves_growth = predictions['Vine1_Leaves'] - current_leaves
    weeks = target_week - current_week
    
    print(f"\nYour vines are predicted to grow {length_growth:.2f} meters")
    print(f"and add {leaves_growth} leaves over {weeks} weeks.")
    print(f"That's about {length_growth/weeks:.3f} meters per week")
    print(f"and {leaves_growth/weeks:.1f} leaves per week.")

if __name__ == "__main__":
    main()