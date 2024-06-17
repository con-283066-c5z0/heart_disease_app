from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import sqlite3

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
