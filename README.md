Titanic Survival Prediction Web Application

This project is an end to end machine learning web application that predicts whether a passenger would have survived the Titanic disaster using a Decision Tree classifier.

The main objective is to demonstrate the complete machine learning lifecycle from data preprocessing and feature engineering to model training and real world deployment using Flask.

Project Overview

The application is trained on the Titanic dataset and allows users to upload a CSV file with passenger information. After uploading, the system predicts the survival outcome for each passenger and displays results on a web interface.

It also allows users to download the prediction results as a new CSV file, which makes it suitable for both academic and practical demonstration purposes.

Features

CSV based batch prediction.
Automatic preprocessing and feature engineering.
Decision Tree based survival prediction.
Survival probability estimation.
Interactive and user friendly web interface.
Downloadable prediction result in CSV format.
50 row sample dataset for testing included.

Machine Learning Approach

The model is a Decision Tree Classifier trained on key Titanic features.

Features used for training

Pclass
Sex
Age
Fare
Embarked
IsAlone

The IsAlone feature is engineered using:

FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0

This improves the predictive performance by capturing family related survival patterns.

Preprocessing Pipeline

The model uses a Scikit Learn pipeline with the following stages:

1. Handling missing values.
Age and Fare are filled using median values.
Categorical features are filled using most frequent values.

2. Encoding categorical variables.
Sex and Embarked are one hot encoded.

3. Feature passthrough.
The engineered IsAlone feature is passed directly into the model.

This preprocessing logic is stored inside the trained model pipeline and automatically applied during prediction.

Web Application Flow

1. User uploads a Titanic style CSV file.

2. System validates if required columns are present.

3. The model preprocesses the data internally.

4. Survival predictions and probabilities are generated.

5. Preview of results is shown on the webpage.

6. User can download the full prediction CSV.

Technologies Used

Python
Flask
Scikit Learn
Pandas
HTML and CSS

Required CSV Format

Your CSV file must contain at least the following columns:

Pclass
Sex
Age
SibSp
Parch
Fare
Embarked

Additional columns like PassengerId, Name, Ticket, Cabin are allowed and will be preserved.

How To Run Locally

1. Clone the repository.

2. Install required dependencies.

3. Run the Flask app.

Example commands:

pip install -r requirements.txt
python app.py


Then open your browser.

Project Structure
titanic_app/
├── app.py
├── model.pkl
├── titanic_sample_template_50rows.csv
├── requirements.txt
└── templates/
    └── index.html

Architecture Overview

The system follows a modular and layered architecture.

1. User Interface Layer
The frontend is built using HTML and CSS and served through Flask templates.
It accepts Titanic style CSV files from users and displays predictions and summary statistics.

2. Application Layer
Flask handles user requests and routes them to appropriate functions.
It manages file uploads, data validation, preprocessing calls and passes data to the machine learning model.

3. Machine Learning Layer
The trained Decision Tree model is stored as a serialized pipeline file (model.pkl).
This includes preprocessing steps like imputation, encoding and feature transformation.

4. Data Processing Layer
Uploaded CSV data is processed using Pandas.
Engineered features like FamilySize and IsAlone are generated before prediction.

5.Output Layer
The final output is displayed in the web interface and also saved into a downloadable CSV file.

6. Data Flow Explanation

User uploads CSV
→ Flask server receives file
→ Data validation and feature engineering
→ Preprocessing pipeline runs
→ Decision Tree makes predictions
→ Predictions returned to UI
→ Results saved and available for download