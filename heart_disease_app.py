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
