import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import kagglehub
import os

def download_dataset():
    """Download Pima Indians Diabetes dataset"""
    print("Downloading dataset...")
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        
        # Find CSV file
        for file in os.listdir(path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, file))
                print(f"Dataset loaded: {df.shape}")
                return df
        
        raise Exception("CSV file not found")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating sample data...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 768
        
        data = {
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
            'BloodPressure': np.random.normal(70, 12, n_samples).clip(0, 122),
            'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 99),
            'Insulin': np.random.normal(80, 100, n_samples).clip(0, 846),
            'BMI': np.random.normal(32, 8, n_samples).clip(0, 67),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples).clip(0.08, 2.42),
            'Age': np.random.randint(21, 81, n_samples)
        }
        
        # Create target with logic
        risk_score = (
            data['Glucose'] * 0.3 +
            data['Age'] * 0.1 +
            data['BMI'] * 0.2 +
            (data['Pregnancies'] > 5) * 20 +
            np.random.normal(0, 15, n_samples)
        )
        
        data['Outcome'] = (risk_score > 50).astype(int)
        
        return pd.DataFrame(data)

def train_model():
    """Train and save model"""
    print("Training model...")
    
    # Load data
    df = download_dataset()
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Save model
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")
    
    # Save feature names
    feature_names = list(X.columns)
    joblib.dump(feature_names, "features.pkl")
    print("Feature names saved")
    
    return model

if __name__ == "__main__":
    train_model()