# SOC-ML-Model-For-Enhanced-Battery-Efficiency
1.Objective

The goal of this project is to develop a machine learning model that can accurately predict the Effective State of Charge (SOC) and to generate relevant Key Performance Indicators (KPIs) that provide insights into battery performance metrics, efficiency, and other operational characteristics.Through detailed data exploration, preprocessing, and model selection, this project aims to build a robust solution that supports performance monitoring and optimization.

2.Over View of DataSet

1.	Fixed Battery Voltage
2.	Portable Battery Voltage
3.	Fixed Battery Current
4.	Portable Battery Current
5.	Motor Status (On/Off)
6.	BCM Battery Selected
7.	Portable Battery Temperatures
8.	Fixed Battery Temperatures

Target Variable: Effective SOC

3.	Data Preprocessing

Steps in Data Preparation

Handling Missing Values: Checked and managed missing values through imputation or removal as necessary.

Encoding Categorical Variables: Transformed the Motor Status feature from categorical (On/Off) to numerical values for model compatibility.

Standardization: Applied standard scaling to ensure that features with differing ranges do not bias the model.

Train-Test Split: Divided the data into training and testing sets to evaluate model performance.

4.	Machine Learning Model Development

4.1. Model Selection and Justification

Random Forest Regressor: Selected for its ability to handle non-linear relationships and feature importance insights.

Hyperparameter Tuning: Optimized parameters such as n_estimators and max_depth to enhance model performance.

4.2. Evaluation Metrics

Mean Squared Error (MSE): Evaluated model error by comparing predictions to actual SOC values.

R2 Score: Measured the goodness of fit, providing an indication of how well the model explains the variance in SOC.

5. Key Performance Indicators (KPIs)

5.1. Defined KPIs

1.	Average SOC Prediction Error: Measures the deviation between predicted and actual SOC values.
2.	Average Battery Range: Indicates the average range covered by the battery based on operational data.
3.	Average Battery Performance (%): Reflects the ratio between actual and predicted charge cycles.
4.	Average Battery Efficiency: Calculated based on the power ratio to understand efficiency levels.
5.	Average Portable Battery Temperature: Provides insights into the temperature variations of portable batteries during operations.
6.	Average Fixed Battery Temperature: Indicates temperature consistency within fixed batteries.

6.	Results

6.1.	Model Performance

MSE: Calculated MSE as a measure of prediction accuracy for SOC.

R2 Score: Provides a summary of model effectiveness.

6.2.	KPI Insights

Battery Efficiency: Average efficiency was observed at approximately X%, indicating reliable energy conversion.

Temperature Monitoring: Noted that temperature levels for both fixed and portable batteries are within optimal operational ranges.

7.	Conclusion

This project successfully developed a machine learning model for predicting Effective SOC and generated KPIs to assess battery performance. The model demonstrates good prediction accuracy and provides actionable insights into battery efficiency, range, and operational health. These KPIs can guide improvements in battery management and inform future optimizations.

