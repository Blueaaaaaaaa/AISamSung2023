import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
train_data = pd.read_csv('Train_samsung.csv')
test_data = pd.read_csv('Test_samsung_noclass.csv')

# EDA
print(train_data.describe())
print(train_data.info())

# Separate features and target
X = train_data.drop('Class', axis=1)
y = train_data['Class']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Function to create pipeline
def create_pipeline(classifier):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

# Evaluate models
for name, classifier in classifiers.items():
    pipeline = create_pipeline(classifier)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f'{name} CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')

# Choose the best model (for this example, let's say it's RandomForest)
best_model = classifiers['RandomForest']

# Create the final pipeline
final_pipeline = create_pipeline(best_model)

# Hyperparameter tuning
param_dist = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [10, 20, 30, 40, 50, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(final_pipeline, param_distributions=param_dist, 
                                   n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)

# Fit the model
random_search.fit(X, y)

# Print best parameters
print("Best parameters:", random_search.best_params_)

# Make predictions on test data
X_test = test_data
predictions = random_search.predict(X_test)

# Convert predictions back to original labels
predictions = le.inverse_transform(predictions)

# Save predictions
result = pd.DataFrame({'prediction': predictions})
result.to_csv('NguyenHaiLam.csv', index=False)