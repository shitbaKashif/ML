# ML
5 different tasks done in internship at Prodigy Infotech. Each folder consists of a separate task containing it's own dataset and different vesrions of that task i.e. if V1 and V2 are present then V1 is the one I wrote first and V2 is the improved version.
## PRODIGY_ML_01
Contains Linear Regression model that predicts house prices based on area, number of bedrooms and bathrooms.
### Housing Price Prediction using Linear Regression
***Overview***

This project involves predicting housing prices using a Linear Regression model. The dataset used contains various features related to houses, such as area, number of bedrooms, bathrooms, stories, and other amenities. The model is enhanced by feature engineering, polynomial features, and categorical encoding to improve performance.

***Requirements***
  - Python 3.6+
  - Pandas
  - NumPy
  - Scikit-learn

***Files***
  - Housing.csv: The dataset containing housing information and prices.
  - V1.ipynb/V2.ipynb: The Python script implementing the Linear Regression model.

***Installation***

Clone the repository or download the files.
Ensure you have the required libraries installed. You can install them using pip:
  - pip install pandas numpy scikit-learn
  - Place the Housing.csv file in the same directory as housing_price_prediction.py.

***Results***
  - *Mean Squared Error (MSE):* A measure of the average squared difference between the actual and predicted values.
  - *R-squared:* A statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables.
  - *Cross-validated R-squared:* The average R-squared value from cross-validation, providing a more reliable estimate of model performance.

***Conclusion***

This project demonstrates how to use Linear Regression for housing price prediction, incorporating feature engineering, polynomial features, and proper handling of categorical data to enhance model performance. The model's performance is evaluated using MSE and R-squared, and cross-validation ensures the results are consistent.


## PRODIGY_ML_02
### K-Means Clustering for Retail Store Customers
***Overview***

This project implements a K-Means Clustering algorithm to group customers of a retail store based on their purchase history. The goal is to identify different customer segments to better understand their behavior and tailor marketing strategies accordingly.

***Dataset***

The dataset used is Mall_Customers.csv, which contains the following columns:
  - *CustomerID:* Unique ID for each customer
  - *Gender:* Gender of the customer (Male/Female)
  - *Age:* Age of the customer
  - *Annual Income (k$):* Annual income of the customer in thousands of dollars
  - *Spending Score (1-100):* Spending score assigned to the customer (1-100)

***Files***
  - *Mall_Customers.csv:* The dataset file
  - *kmeans_clustering.py:* The Python script implementing the K-Means Clustering algorithm

***Dependencies***
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  *You can install the required libraries using the following command:*
    - !pip install pandas numpy matplotlib seaborn scikit-learn

***Usage***

Ensure that the Mall_Customers.csv file is in the same directory as the V1.ipynb/V2.ipynb script.
*The script will:*
  - Load and preprocess the dataset
  - Determine the optimal number of clusters using the Elbow Method
  - Apply the K-Means Clustering algorithm
  - Visualize the clusters
  - Save the clustered data to Result.csv

***Conclusion***

This project demonstrates how to use the K-Means Clustering algorithm to segment customers based on their purchase history. By identifying different customer segments, businesses can tailor their marketing strategies and improve customer satisfaction.

## Image Classification of Cats and Dogs using SVM

***Introduction***

SVM is a powerful supervised learning algorithm used for classification tasks. In this project, we leverage a pre-trained VGG16 model to extract features from images of cats and dogs, which are then used to train an SVM for classification. This approach improves accuracy by utilizing the rich feature representations learned by VGG16 from the ImageNet dataset.

***Dataset***

The dataset used in this project consists of images of cats and dogs. Ensure the dataset is organized in two directories:
  - *./train/cats:* Contains images of cats.
  - *./train/dogs:* Contains images of dogs.

***Installation***

To run this project, you need to have Python installed along with the following libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow
  - opencv-python
You can install the required libraries using pip:
  - !pip install numpy pandas scikit-learn matplotlib seaborn tensorflow opencv-python

***Usage***
  - Clone the repository or download the dataset from kaggle (https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?select=train).
  - Create a Jupiter Notebook.
  - Run the notebook to perform SVM classification and visualize the results:
      - V1.ipynb/V2.ipynb

***Results***

The script will output the following:
  - Accuracy of the SVM model on the test set.
  - Classification report showing precision, recall, and f1-score for each class.
  - Confusion matrix visualized using a heatmap.

***Further Improvements***
  - *Hyperparameter Tuning:* Use grid search or random search to find the best hyperparameters for the SVM model.
  - *Data Augmentation:* Apply data augmentation techniques such as rotation, flipping, and zooming to increase the variability of the training data.
  - *Experiment with Different CNNs:* Try other pre-trained models like ResNet50, InceptionV3, or MobileNet for feature extraction.
  - *Regularization:* Adjust regularization parameters to prevent overfitting.
