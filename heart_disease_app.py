from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import sqlite3
import joblib
import streamlit as st

#Name of table
table_name = "heart_table"

#Reopened the connection to the SQLite database
connection = sqlite3.connect("heart.db")

#Get data from db to df
query = f"SELECT * FROM {table_name};"
df = pd.read_sql(query, connection)

#Closed the connection
connection.close()

#Defined features and target
X = df.drop(columns='target')
y = df['target']

#Identified numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'int64']).columns.difference(numerical_cols)

#Preprocessed for numerical data: imputer used for missing values and scaler for removing the mean scaling to unit varience.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#Preprocessed for categorical data: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Applied transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#Defined and trained the models
models = {
    'Logistic Regression': Pipeline(steps=[
        ('classifier', LogisticRegression())
    ]),
    'Naive Bayes': Pipeline(steps=[
        ('classifier', GaussianNB())
    ]),
    'Random Forest': Pipeline(steps=[
        ('classifier', RandomForestClassifier())
    ])
}

best_model = None
best_accuracy = 0

for model_name, model_pipeline in models.items():
    #Trained the model
    model_pipeline.fit(X_train, y_train)
    #Predicted the test set
    y_pred = model_pipeline.predict(X_test)
    #Evaluated the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_pipeline

#Indicated the best model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print(f'Best model: {best_model.steps[-1][1]} with accuracy: {best_accuracy}')

#Loaded the trained Naive Bayes model and processesor
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

#Heart disease app
st.title("Heart Disease Prediction")

#Requested the users input data
st.header("Please Enter The Patient's Details")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.number_input("Number of Major Vessels Colored by Flourosopy (ca)", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

#DF of user input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

#Preprocessed the input data
input_data_processed = preprocessor.transform(input_data)

#Used Naive Bayes model
prediction = model.predict(input_data_processed)[0]

#Displayed the prediction result
st.header("Prediction Result")
if prediction == 1:
    st.write("The patient is LIKELY to have heart disease. Further tests or treatment may be necessary.")
else:
    st.write("The patient is UNLIKELY to have heart disease.")
