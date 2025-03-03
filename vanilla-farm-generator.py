import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_vanilla_farm_data():
    # Initialize lists to store our data
    dates = []
    weeks = []
    temperatures = []
    humidities = []
    light_levels = []
    
    # Initialize vine measurements
    vine_lengths = {
        'vine1': {'start': 0.50, 'end': 13.58},
        'vine2': {'start': 0.63, 'end': 12.69},
        'vine3': {'start': 0.62, 'end': 11.73},
        'vine4': {'start': 0.58, 'end': 13.82}
    }
    
    vine_leaves = {
        'vine1': {'start': 4, 'end': 173},
        'vine2': {'start': 5, 'end': 159},
        'vine3': {'start': 4, 'end': 180},
        'vine4': {'start': 7, 'end': 167}
    }
    
    # Lists to store vine measurements
    vine1_lengths, vine2_lengths, vine3_lengths, vine4_lengths = [], [], [], []
    vine1_leaves, vine2_leaves, vine3_leaves, vine4_leaves = [], [], [], []
    
    # Set start date
    start_date = datetime(2022, 1, 1)
    
    # Generate data for 131 weeks (3 years)
    for week in range(131):
        # Calculate current date
        current_date = start_date + timedelta(weeks=week)
        dates.append(current_date.strftime('%Y-%m-%d'))
        weeks.append(week + 1)
        
        # Generate environmental data with seasonal variations
        # Temperature: Base 24°C with seasonal variations
        month = current_date.month
        if month in [3, 4]:  # Warmer in March-April
            temp = np.random.uniform(24.5, 26.5)
        else:
            temp = np.random.uniform(22.5, 25.5)
        temperatures.append(round(temp, 1))
        
        # Humidity: Higher during monsoon season
        if month in [10, 11, 12]:  # Monsoon season
            humidity = np.random.uniform(75, 85)
        else:
            humidity = np.random.uniform(70, 80)
        humidities.append(round(humidity, 1))
        
        # Light levels: Lower during monsoon
        if month in [10, 11, 12]:
            light = np.random.uniform(25000, 30000)
        else:
            light = np.random.uniform(28000, 35000)
        light_levels.append(round(light, 0))
        
        # Calculate growth progress (0 to 1)
        progress = week / 130
        
        # Generate vine lengths with slight variations
        for vine_num, measurements in vine_lengths.items():
            growth = measurements['start'] + (measurements['end'] - measurements['start']) * progress
            # Add small random variation
            growth += np.random.uniform(-0.1, 0.1)
            growth = max(growth, measurements['start'])  # Ensure we don't go below start value
            
            if vine_num == 'vine1':
                vine1_lengths.append(round(growth, 2))
            elif vine_num == 'vine2':
                vine2_lengths.append(round(growth, 2))
            elif vine_num == 'vine3':
                vine3_lengths.append(round(growth, 2))
            else:
                vine4_lengths.append(round(growth, 2))
        
        # Generate leaf counts with slight variations
        for vine_num, leaf_counts in vine_leaves.items():
            leaves = leaf_counts['start'] + (leaf_counts['end'] - leaf_counts['start']) * progress
            # Add small random variation and round to nearest integer
            leaves = round(leaves + np.random.uniform(-1, 1))
            leaves = max(leaves, leaf_counts['start'])  # Ensure we don't go below start value
            
            if vine_num == 'vine1':
                vine1_leaves.append(leaves)
            elif vine_num == 'vine2':
                vine2_leaves.append(leaves)
            elif vine_num == 'vine3':
                vine3_leaves.append(leaves)
            else:
                vine4_leaves.append(leaves)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Week': weeks,
        'Temperature(C)': temperatures,
        'Humidity(%)': humidities,
        'LightLevel(lux)': light_levels,
        'Vine1_Length(m)': vine1_lengths,
        'Vine2_Length(m)': vine2_lengths,
        'Vine3_Length(m)': vine3_lengths,
        'Vine4_Length(m)': vine4_lengths,
        'Vine1_Leaves': vine1_leaves,
        'Vine2_Leaves': vine2_leaves,
        'Vine3_Leaves': vine3_leaves,
        'Vine4_Leaves': vine4_leaves
    })
    
    # Save to CSV
    df.to_csv('vanilla_farm_data.csv', index=False)
    print("Dataset has been generated and saved as 'vanilla_farm_data.csv'")

# Run the function
if __name__ == "__main__":
    generate_vanilla_farm_data()
