Project Title: Traffic Accident Severity Prediction

Objective:
    Predict the severity of traffic accidents based on environmental, road, and weather conditions using machine learning.

Dataset:
    US Accidents Dataset (Kaggle)
    Source : https://www.kaggle.com/datasets/sasikumarg/us-accidents
    File name: US_Accidents.csv

Insights :
    
    -> dataset contains 1 million + records with weather, road and environmental features
    
    -> After data cleaning and feature engineering
        -> There were no rows with more than 50% null values
        -> Time based features such as day, month, year, weekdy are extracted from Start_Time column.
        -> Encoded Categorical Variables
        -> Numerical Features were Scaled
    
    -> Logistic Regression suffered from long training time and poor minority class recall.
    
    -> From Random Forest and XGBoost feature importance plots:
        - Distance(mi) and Visibility(mi) were strong predictors — longer distances and poor visibility increased severity.
        - Weather_Condition, Wind_Speed(mph), and Precipitation(in) also contributed significantly, indicating that weather has a clear impact on severity.
        - Time of day (Hour) showed a pattern: morning and evening rush hours had higher severity counts.
        - Location-based features like City and State influenced the model but may also encode reporting or infrastructure differences.

    -> Real-World Applications
        - Traffic Control
        - Alert driver regarding weather condition related severity
        - Location Based Planning for maintenance of accident prone locations