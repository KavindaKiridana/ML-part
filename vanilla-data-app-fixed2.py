import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# Title and description
st.title("Vanilla Vine Growth Predictor")
st.write("""
This application predicts how your vanilla vines will grow over time.
Enter your current measurements and see what to expect in the future!
""")

# Add the "View Farm" button at the top of the application
# NEW CODE: Added button that opens a new tab with the portfolio website when clicked
if st.button("View Farm"):
    # Open URL in a new tab (this works when the app is run locally)
    # Note: In deployed Streamlit apps, we need to use JavaScript for redirection
    st.markdown(f'<meta http-equiv="refresh" content="0;url={"https://kavindakiridana.github.io/myportfolio/"}">', unsafe_allow_html=True)
    st.success("Redirecting to farm website...")

# Function to load or train models
@st.cache_resource
def get_models(data_file='vanilla_data.txt'):
    model_file = 'vanilla_growth_models.pkl'
    
    # Check if models exist
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        # Train new models if file doesn't exist
        st.info("Training models for the first time. This may take a moment...")
        # Load data
        df = pd.read_csv(data_file, sep='\t')
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Year'] = df['Date'].dt.year - df['Date'].dt.year.min() + 1
        
        # List of vine names for easier iteration
        vine_names = ['Vine1', 'Vine2', 'Vine3', 'Vine4']
        
        # Prepare features - FIXED: using "Week" instead of "WeekNo"
        features = ['Week', 'Temperature(C)', 'Humidity(%)', 'LightLevel(lux)', 'Year']
        
        # Dictionary to store models
        models = {}
        
        for vine in vine_names:
            # Create models for length
            X = df[features]
            y_length = df[f'{vine}_Length(m)']
            
            # Train model for length
            model_length = RandomForestRegressor(n_estimators=100, random_state=42)
            model_length.fit(X, y_length)
            
            # Create models for leaves
            y_leaves = df[f'{vine}_Leaves']
            
            # Train model for leaves
            model_leaves = RandomForestRegressor(n_estimators=100, random_state=42)
            model_leaves.fit(X, y_leaves)
            
            # Store models
            models[f'{vine}_Length'] = model_length
            models[f'{vine}_Leaves'] = model_leaves
        
        # Save models
        joblib.dump(models, model_file)
        return models

# Load or train models
try:
    models = get_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar for inputs
st.sidebar.header("Your Vine Measurements")
st.sidebar.markdown("Enter the current details of your vanilla vine:")

current_week = st.sidebar.number_input("Current Week Number", min_value=1, max_value=156, value=1)
current_length = st.sidebar.number_input("Current Vine Length (meters)", min_value=0.1, max_value=15.0, value=0.5, step=0.1)
current_leaves = st.sidebar.number_input("Current Number of Leaves", min_value=1, max_value=200, value=5)

st.sidebar.header("Target Prediction")
target_week = st.sidebar.slider("Predict for Week Number", min_value=current_week+1, max_value=156, value=min(current_week+52, 156))

st.sidebar.header("Environmental Factors (Optional)")
temperature = st.sidebar.slider("Average Temperature (°C)", min_value=20.0, max_value=30.0, value=24.5, step=0.1)
humidity = st.sidebar.slider("Average Humidity (%)", min_value=60.0, max_value=90.0, value=75.0, step=0.1)
light_level = st.sidebar.slider("Average Light Level (lux)", min_value=25000, max_value=35000, value=30000, step=100)

# Function to predict growth
def predict_growth(initial_week, target_week, temp, humidity, light):
    # Calculate the year based on weeks
    initial_year = initial_week // 52 + 1
    target_year = target_week // 52 + 1
    
    # Create input features - FIXED: using "Week" instead of "WeekNo"
    input_features = pd.DataFrame({
        'Week': [target_week],
        'Temperature(C)': [temp],
        'Humidity(%)': [humidity],
        'LightLevel(lux)': [light],
        'Year': [target_year]
    })
    
    # Make predictions for each vine
    predictions = {}
    for vine in ['Vine1', 'Vine2', 'Vine3', 'Vine4']:
        length_model = models[f'{vine}_Length']
        leaves_model = models[f'{vine}_Leaves']
        
        predicted_length = length_model.predict(input_features)[0]
        predicted_leaves = leaves_model.predict(input_features)[0]
        
        predictions[f'{vine}_Length'] = predicted_length
        predictions[f'{vine}_Leaves'] = int(round(predicted_leaves))
    
    return predictions

# Generate growth projection
if st.button("Generate Growth Projection"):
    # Make predictions
    predictions = predict_growth(current_week, target_week, temperature, humidity, light_level)
    
    # Display results
    st.header("Predicted Growth Results")
    
    # Show weeks and time span
    weeks_difference = target_week - current_week
    years_difference = weeks_difference / 52
    
    st.write(f"Projection from Week {current_week} to Week {target_week}")
    st.write(f"Time span: {weeks_difference} weeks ({years_difference:.1f} years)")
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    # Display the results in a table
    results_df = pd.DataFrame({
        'Vine': ['Vine 1', 'Vine 2', 'Vine 3', 'Vine 4'],
        'Predicted Length (m)': [
            predictions['Vine1_Length'],
            predictions['Vine2_Length'],
            predictions['Vine3_Length'],
            predictions['Vine4_Length']
        ],
        'Predicted Leaves': [
            predictions['Vine1_Leaves'],
            predictions['Vine2_Leaves'],
            predictions['Vine3_Leaves'],
            predictions['Vine4_Leaves']
        ]
    })
    
    with col1:
        st.dataframe(results_df)
    
    # Display a bar chart
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot lengths
        bar_positions = np.arange(4)
        bar_width = 0.35
        
        lengths = [
            predictions['Vine1_Length'],
            predictions['Vine2_Length'],
            predictions['Vine3_Length'],
            predictions['Vine4_Length']
        ]
        
        ax.bar(bar_positions, lengths, bar_width, label='Length (m)')
        
        # Add labels and title
        ax.set_xlabel('Vine')
        ax.set_ylabel('Predicted Length (m)')
        ax.set_title('Predicted Vine Length')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(['Vine 1', 'Vine 2', 'Vine 3', 'Vine 4'])
        
        st.pyplot(fig)
    
    # Growth comparison
    st.header("Growth Comparison")
    
    growth_df = pd.DataFrame({
        'Vine': ['Vine 1', 'Vine 2', 'Vine 3', 'Vine 4'],
        'Current Length (m)': [current_length, current_length, current_length, current_length],
        'Predicted Length (m)': [
            predictions['Vine1_Length'],
            predictions['Vine2_Length'],
            predictions['Vine3_Length'],
            predictions['Vine4_Length']
        ],
        'Current Leaves': [current_leaves, current_leaves, current_leaves, current_leaves],
        'Predicted Leaves': [
            predictions['Vine1_Leaves'],
            predictions['Vine2_Leaves'],
            predictions['Vine3_Leaves'],
            predictions['Vine4_Leaves']
        ]
    })
    
    growth_df['Length Growth (m)'] = growth_df['Predicted Length (m)'] - growth_df['Current Length (m)']
    growth_df['Leaves Growth'] = growth_df['Predicted Leaves'] - growth_df['Current Leaves']
    
    st.dataframe(growth_df)
    
    # Additional insights
    st.header("Growth Insights")
    
    avg_length_growth = growth_df['Length Growth (m)'].mean()
    avg_leaves_growth = growth_df['Leaves Growth'].mean()
    
    st.write(f"On average, your vanilla vines are predicted to grow {avg_length_growth:.2f} meters and add {int(avg_leaves_growth)} leaves over the selected time period.")
    
    growth_rate_per_week = avg_length_growth / weeks_difference
    leaves_rate_per_week = avg_leaves_growth / weeks_difference
    
    st.write(f"Weekly growth rate: {growth_rate_per_week:.3f} meters/week")
    st.write(f"Weekly leaves addition rate: {leaves_rate_per_week:.2f} leaves/week")
    
    # Recommendations based on predictions
    st.header("Recommendations")
    recommendations = [
        "Ensure adequate support structures as vines grow longer",
        f"Plan for approximately {int(avg_length_growth * 100)} cm of vertical growth space",
        "Monitor leaf health regularly as the plant develops new foliage",
        "Adjust fertilization schedule based on the predicted growth rate"
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")
        
    st.info("Note: These predictions are based on historical data patterns. Actual growth may vary based on care, environment, and vine genetics.")
