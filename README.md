# ML
5 different tasks done in internship at Prodigy Infotech. Each folder consists of a separate task containing it's own dataset and different vesrions of that task i.e. if V1 and V2 are present then V1 is the one I wrote first and V2 is the improved version.
# PRODIGY_ML_01
Contains Linear Regression model that predicts house prices based on area, number of bedrooms and bathrooms.
## Housing Price Prediction using Linear Regression
***Overview***
This project involves predicting housing prices using a Linear Regression model. The dataset used contains various features related to houses, such as area, number of bedrooms, bathrooms, stories, and other amenities. The model is enhanced by feature engineering, polynomial features, and categorical encoding to improve performance.

***Requirements***
  - Python 3.6+
  - Pandas
  - NumPy
  - Scikit-learn

***Files***
  - Housing.csv: The dataset containing housing information and prices.
  - V1.py/V2.py: The Python script implementing the Linear Regression model.

***Installation***
Clone the repository or download the files.
Ensure you have the required libraries installed. You can install them using pip:
  - pip install pandas numpy scikit-learn
  - Place the Housing.csv file in the same directory as housing_price_prediction.py.

***Usage***
Run the housing_price_prediction.py script to train the model and make predictions:
  - python V2.py or python V1.py

***Results***
  - *Mean Squared Error (MSE):* A measure of the average squared difference between the actual and predicted values.
  - *R-squared:* A statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables.
  - *Cross-validated R-squared:* The average R-squared value from cross-validation, providing a more reliable estimate of model performance.

***Conclusion***
This project demonstrates how to use Linear Regression for housing price prediction, incorporating feature engineering, polynomial features, and proper handling of categorical data to enhance model performance. The model's performance is evaluated using MSE and R-squared, and cross-validation ensures the results are consistent.
