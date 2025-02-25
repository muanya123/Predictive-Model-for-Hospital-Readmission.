# Diabetic Patient Readmission Prediction Project #

### Overview ###
This project is focused on predicting hospital readmission for diabetic patients using machine learning techniques. The objective is to build a robust classification model that determines whether a patient will be readmitted within 30 days (readmitted = 1) or not (readmitted = 0). The model leverages structured numerical data, categorical features, and textual clinical notes to generate a comprehensive feature set.

### Key Features ###

Data Preprocessing:

1. Missing Values: Missing values in clinical notes are handled by filling with a 		default value (e.g., an empty string or 'nan' for categoriy
2. Column Dropping: Unwanted columns are removed to focus on relevant features.

### Feature Engineering: ### 

1. TF-IDF Transformation: Clinical notes (free text) are converted into numerical 		features using a TF-IDF vectorizer, limiting the vocabulary to the top 100 terms.
2. One-Hot Encoding: Categorical columns such as race, gender, and medication types 	are transformed into binary columns to allow the model to process them.
3. Structured Data Integration: Numeric features like time_in_hospital, 				num_lab_procedures, etc., are combined with encoded categorical and text-based 		features into one final dataset.

### Model Training: ###

A Random Forest Classifier is trained on the processed data using specified hyperparameters (e.g., 1000 estimators, entropy criterion, and a maximum depth of 25) to predict patient readmission outcomes.

### Test Data Prediction: ###

The same preprocessing and feature engineering pipeline is applied to the test dataset.
The trained model is used to predict outcomes on the test data, and the results are aggregated to analyze the distribution of predictions.


## Project Structure ##

	├── data/
	│   ├── diabetic_data.csv         # Training dataset
	│   └── diabetic_data_test.csv    # Test dataset
	├── notebooks/                    # Jupyter notebooks for exploration and model 	training
	├── src/
	│   ├── preprocessing.py          # Data cleaning and feature engineering 			functions
	│   ├── model_training.py         # Model training and evaluation code
	│   └── predict.py                # Code to preprocess test data and generate 		predictions
	├── README.md                     # This file
	└── requirements.txt              # List of dependencies and Python packages


## How It Works ##

1. #### Data Loading and Cleaning: ####
	The datasets are loaded into pandas DataFrames. Missing values are handled, and 	irrelevant columns are dropped to ensure the data is clean and ready for 			analysis.

2. #### Feature Engineering: #### 

TF-IDF Transformation: Converts clinical text notes into numerical features.

One-Hot Encoding: Converts categorical variables into a numeric format that can be interpreted by the machine learning algorithm.

Feature Combination: Structured numeric features are concatenated with the transformed text and categorical features to form a final feature set.

3. #### Model Training and Evaluation: #### 

The final training data is split into training and validation sets.

A Random Forest model is trained using the training set and evaluated on the validation set.

Accuracy metrics are computed to assess model performance.

4. #### Prediction on Test Data: ####

The test dataset undergoes the same preprocessing and feature engineering as the training data.
The trained model is then used to predict readmission outcomes.
Predictions are appended to the test dataset, and the distribution of predicted classes is summarized.


## How to Run ##
1. #### Install Dependencies: ####
	Ensure you have Python 3.7+ installed and run:
	
 		bash
		pip install -r requirements.txt
 
2. #### Preprocessing and Model Training: ####
	Run the model training script:

		bash
		python src/model_training.py

	This will load the training data, preprocess it, train the Random Forest 			Classifier, and output accuracy metrics.

3. #### Test Data Prediction: ####
	Once the model is trained, run the prediction script:

		bash
		python src/predict.py

	This script preprocesses the test data using the same pipeline and generates 		predictions, printing a summary count of readmission predictions.

## Important Notes ##

#### Consistency in Preprocessing: ####
For accurate predictions, the preprocessing steps (TF-IDF transformation and one hot encoding) must be identical between training and test datasets. In production, ensure you reuse the fitted transformers rather than refitting on the test data to maintain consistency in feature names and order.

#### Model Evaluation: ####
The project currently evaluates the model based on training and validation accuracy. Further tuning and cross validation may improve model performance and generalizability.

#### Future Improvements: ####
Future work could include hyperparameter tuning, incorporating additional feature engineering, or experimenting with alternative models to improve prediction accuracy.

#### Conclusion ####
This project demonstrates a complete end to end machine learning pipeline from data cleaning and feature engineering to model training and prediction for predicting diabetic patient readmission. It illustrates the importance of consistency in data processing and the practical application of ensemble methods for healthcare analytics.

For any further questions or contributions, please feel free to reach out or open an issue on the repository. Enjoy exploring the project!


[Readme.docx](https://github.com/user-attachments/files/18952554/Readme.docx)
