import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
import pickle
from car_data_prep import prepare_data   # Ensure car_data_prep is in the same directory or adjust the import path

# Load the dataset
file_name = 'dataset.csv'
data = pd.read_csv(file_name)

# Apply the data_prep function to the dataset
prepared_data = prepare_data(data) 

# Separate features and target variable
X = prepared_data.drop('Price', axis=1)
y = prepared_data['Price']


# Identify numerical and categorical features
numerical_types = ['int','int16','int32','int64','float','float16','float32','float64']
numerical_features = X.select_dtypes(include= numerical_types).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define numerical pipeline
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])



# Define categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numerical and categorical pipelines into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Define the ElasticNet model
elastic_net = ElasticNet()

# Create the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', elastic_net)
])

# Define the best parameters (from previous tuning or assumptions)
best_alpha = 0.001
best_l1_ratio = 0.1

# Create the final ElasticNet model with the best parameters
final_elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)

# Update the pipeline with the final model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_elastic_net)
])

# Train the final pipeline on the entire dataset
final_pipeline.fit(X, y)

# Save the trained model
pickle.dump(final_pipeline, open("elasticNet_model.pkl", "wb"))

print("Model training completed and saved as 'elasticNet_model.pkl'.")
