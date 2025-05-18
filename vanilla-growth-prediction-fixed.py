import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

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
    # Features will be: Week, Temperature, Humidity, LightLevel, Year, and initial measurements
    # For each vine, we'll create separate models
    
    # List of vine names for easier iteration
    vine_names = ['Vine1', 'Vine2', 'Vine3', 'Vine4']
    
    # Prepare features - FIXED: using "Week" instead of "WeekNo"
    features = ['Week', 'Temperature(C)', 'Humidity(%)', 'LightLevel(lux)', 'Year']
    
    # Dictionary to store models for each vine and measurement type
    models = {}
    
    for vine in vine_names:
        # Create models for length
        X = df[features]
        y_length = df[f'{vine}_Length(m)']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_length, test_size=0.2, random_state=42)
        
        # Train model for length
        model_length = RandomForestRegressor(n_estimators=100, random_state=42)
        model_length.fit(X_train, y_train)
        
        # Create models for leaves
        y_leaves = df[f'{vine}_Leaves']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_leaves, test_size=0.2, random_state=42)
        
        # Train model for leaves
        model_leaves = RandomForestRegressor(n_estimators=100, random_state=42)
        model_leaves.fit(X_train, y_train)
        
        # Store models
        models[f'{vine}_Length'] = model_length
        models[f'{vine}_Leaves'] = model_leaves
        
        # Evaluate models
        print(f"Model for {vine} Length:")
        y_pred = model_length.predict(X_test)
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        
        print(f"Model for {vine} Leaves:")
        y_pred = model_leaves.predict(X_test)
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print("-" * 50)
    
    return models

# Step 3: Visualize growth trends
def visualize_growth(df):
    plt.figure(figsize=(12, 10))
    
    # Plot length growth over time - FIXED: using "Week" instead of "WeekNo"
    plt.subplot(2, 1, 1)
    plt.plot(df['Week'], df['Vine1_Length(m)'], label='Vine 1')
    plt.plot(df['Week'], df['Vine2_Length(m)'], label='Vine 2')
    plt.plot(df['Week'], df['Vine3_Length(m)'], label='Vine 3')
    plt.plot(df['Week'], df['Vine4_Length(m)'], label='Vine 4')
    plt.xlabel('Week Number')
    plt.ylabel('Length (m)')
    plt.title('Vanilla Vine Length Growth Over Time')
    plt.legend()
    
    # Plot leaves growth over time - FIXED: using "Week" instead of "WeekNo"
    plt.subplot(2, 1, 2)
    plt.plot(df['Week'], df['Vine1_Leaves'], label='Vine 1')
    plt.plot(df['Week'], df['Vine2_Leaves'], label='Vine 2')
    plt.plot(df['Week'], df['Vine3_Leaves'], label='Vine 3')
    plt.plot(df['Week'], df['Vine4_Leaves'], label='Vine 4')
    plt.xlabel('Week Number')
    plt.ylabel('Number of Leaves')
    plt.title('Vanilla Vine Leaves Growth Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vanilla_growth_trends.png')
    plt.close()

# Step 4: Create a function to predict future growth
def predict_future_growth(models, initial_week, initial_length, initial_leaves, 
                          target_week, temp=24.5, humidity=75.0, light=30000):
    # Calculate the number of years based on weeks (approximate)
    initial_year = initial_week // 52 + 1
    target_year = target_week // 52 + 1
    
    # Create a dataframe for the input features - FIXED: using "Week" instead of "WeekNo"
    input_df = pd.DataFrame({
        'Week': [target_week],
        'Temperature(C)': [temp],
        'Humidity(%)': [humidity],
        'LightLevel(lux)': [light],
        'Year': [target_year]
    })
    
    # Make predictions for each vine
    predictions = {}
    for vine in ['Vine1', 'Vine2', 'Vine3', 'Vine4']:
        # Predict length
        length_model = models[f'{vine}_Length']
        predicted_length = length_model.predict(input_df)[0]
        
        # Predict leaves
        leaves_model = models[f'{vine}_Leaves']
        predicted_leaves = leaves_model.predict(input_df)[0]
        
        # Store predictions
        predictions[f'{vine}_Length'] = predicted_length
        predictions[f'{vine}_Leaves'] = predicted_leaves
    
    return predictions

# Step 5: Save models for later use
def save_models(models, filename='vanilla_growth_models.pkl'):
    joblib.dump(models, filename)
    print(f"Models saved to {filename}")

# Step 6: Load saved models
def load_models(filename='vanilla_growth_models.pkl'):
    return joblib.load(filename)

# Step 7: Create a simple command-line application
def run_prediction_app():
    print("\n===== Vanilla Vine Growth Prediction Tool =====\n")
    
    # Load models
    try:
        models = load_models()
        print("Models loaded successfully.\n")
    except:
        print("Error loading models. Please run training first.")
        return
    
    # Get user input
    try:
        initial_week = int(input("Enter current week number (1-156): "))
        initial_length = float(input("Enter current vine length in meters: "))
        initial_leaves = int(input("Enter current number of leaves: "))
        target_week = int(input("Enter target week number to predict for: "))
        
        # Optional environmental factors
        print("\nEnter environmental factors (or press Enter for defaults):")
        temp_input = input("Average temperature in °C (default: 24.5): ")
        humidity_input = input("Average humidity in % (default: 75.0): ")
        light_input = input("Average light level in lux (default: 30000): ")
        
        temp = float(temp_input) if temp_input else 24.5
        humidity = float(humidity_input) if humidity_input else 75.0
        light = float(light_input) if light_input else 30000
        
        # Make prediction
        predictions = predict_future_growth(
            models, initial_week, initial_length, initial_leaves, 
            target_week, temp, humidity, light
        )
        
        # Display results
        print("\n===== Predicted Growth =====")
        print(f"From Week {initial_week} to Week {target_week}:\n")
        
        for vine in ['Vine1', 'Vine2', 'Vine3', 'Vine4']:
            print(f"{vine}:")
            print(f"  Predicted Length: {predictions[f'{vine}_Length']:.2f} meters")
            print(f"  Predicted Leaves: {int(round(predictions[f'{vine}_Leaves']))}")
            print()
            
        print("Note: These predictions are based on the patterns in your dataset.")
        print("Environmental factors and care practices may affect actual growth.")
        
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the entire process
def main():
    print("Vanilla Vine Growth Prediction System")
    print("=====================================")
    
    # Step 1: Ask user for data file
    file_path = input("Enter the path to your data file (or press Enter for default 'vanilla_data.txt'): ")
    if not file_path:
        file_path = 'vanilla_data.txt'
    
    try:
        # Load and process data
        print(f"Loading data from {file_path}...")
        df = load_data(file_path)
        print(f"Loaded {len(df)} records.")
        
        # Visualize the data
        print("Generating growth trend visualization...")
        visualize_growth(df)
        print("Visualization saved as 'vanilla_growth_trends.png'")
        
        # Train models
        print("Training prediction models...")
        models = prepare_data(df)
        
        # Save models
        save_models(models)
        
        # Run the prediction application
        run_again = 'y'
        while run_again.lower() == 'y':
            run_prediction_app()
            run_again = input("\nMake another prediction? (y/n): ")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()