# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Define the size of the dataset
num_samples = 1000

# Generate synthetic data for battery characteristics
data = {
    'Fixed Battery Voltage': np.random.normal(loc=350, scale=15, size=num_samples),  # around 350V
    'Portable Battery Voltage': np.random.normal(loc=15, scale=3, size=num_samples),  # around 15V
    'Portable Battery Current': np.random.normal(loc=3, scale=1, size=num_samples),   # around 3A
    'Fixed Battery Current': np.random.normal(loc=15, scale=2, size=num_samples),     # around 15A
    'Motor Status': np.random.choice(['On', 'Off'], num_samples),
    'BCM Battery Selected': np.random.choice([0, 1], num_samples),
    'Portable Battery Temperatures': np.random.normal(loc=30, scale=5, size=num_samples),  # around 30°C
    'Fixed Battery Temperatures': np.random.normal(loc=40, scale=5, size=num_samples),     # around 40°C
    'USD/CNY': np.random.uniform(6.0, 7.0, num_samples),  # exchange rate range
    'Effective SOC': np.clip(np.random.normal(loc=50, scale=20, size=num_samples), 0, 100) # SOC % between 0-100
}

# Create DataFrame
df = pd.DataFrame(data)

# Display first few rows and data summary
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
# Check for missing values
print(df.isnull().sum())

# Encode categorical features
df['Motor Status'] = LabelEncoder().fit_transform(df['Motor Status'])

# Standardize the numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(['Effective SOC'], axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['Effective SOC'] = df['Effective SOC']

# Split the data
X = df_scaled.drop(['Effective SOC'], axis=1)
y = df_scaled['Effective SOC']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Calculate the prediction error as a KPI
df_results = pd.DataFrame({'Actual SOC': y_test, 'Predicted SOC': y_pred})
df_results['SOC Difference'] = df_results['Predicted SOC'] - df_results['Actual SOC']

# Average Prediction Error
average_soc_error = df_results['SOC Difference'].mean()

# Define Battery Range and Battery Performance (simulated)
# Since we don't have actual values, we'll calculate hypothetical values
# Here we're using the 'Fixed Battery Voltage' as a proxy for battery range
average_battery_range = df['Fixed Battery Voltage'].mean()  # Replace with your actual battery range logic

# Define Battery Performance (ensuring actual_charge_cycles and predicted_charge_cycles are calculated)
df['actual_charge_cycles'] = np.random.randint(1, 100, size=len(df))  # Dummy data
df['predicted_charge_cycles'] = np.random.randint(1, 100, size=len(df))  # Dummy data
df['battery_performance'] = (df['actual_charge_cycles'] / df['predicted_charge_cycles']) * 100
average_battery_performance = df['battery_performance'].mean()

# KPI Summary
kpi_summary = {
    "Average SOC Prediction Error": average_soc_error,
    "Average Battery Range": average_battery_range,
    "Average Battery Performance (%)": average_battery_performance,
}

for kpi, value in kpi_summary.items():
    print(f"{kpi}: {value:.2f}\n")


# Visualization of the prediction error
plt.figure(figsize=(10, 6))

# Plot histogram
sns.histplot(df_results['SOC Difference'], bins=30, color='blue', label='SOC Prediction Error', alpha=0.6)

# Plot KDE line
sns.kdeplot(df_results['SOC Difference'], color='navy', label='KDE of SOC Prediction Error', linewidth=2)

plt.title('Distribution of SOC Prediction Error')
plt.xlabel('SOC Prediction Error')
plt.ylabel('Frequency')

# Add a vertical line at zero error
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')

# Add a legend to clarify the lines
plt.legend()

# Show the plot
plt.show()

# Additional KPI Calculation

# Calculate Battery Efficiency
df['Battery Efficiency'] = (df['Fixed Battery Voltage'] * df['Fixed Battery Current']) / \
                           (df['Portable Battery Voltage'] * df['Portable Battery Current'])

# Calculate Average Portable Battery Temperature
average_portable_battery_temperature = df['Portable Battery Temperatures'].mean()

# Calculate Average Fixed Battery Temperature
average_fixed_battery_temperature = df['Fixed Battery Temperatures'].mean()

# Calculate Power Ratio
df['Power Ratio'] = (df['Fixed Battery Voltage'] * df['Fixed Battery Current']) / \
                    (df['Portable Battery Voltage'] * df['Portable Battery Current'])

# Calculate Average Power Ratio
average_power_ratio = df['Power Ratio'].mean()

# Summarizing the additional KPIs
additional_kpi_summary = {
    "Average Battery Efficiency": df['Battery Efficiency'].mean(),
    "Average Portable Battery Temperature": average_portable_battery_temperature,
    "Average Fixed Battery Temperature": average_fixed_battery_temperature,
    "Average Power Ratio": average_power_ratio
}

for kpi, value in additional_kpi_summary.items():
    print(f"{kpi}: {value:.2f}\n")



