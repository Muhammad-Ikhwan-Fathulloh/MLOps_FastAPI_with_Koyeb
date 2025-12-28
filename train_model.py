"""
Train model tanpa download dari kaggle
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_diabetes_data():
    """Create synthetic diabetes data based on Pima Indians statistics"""
    np.random.seed(42)
    n_samples = 768
    
    # Berdasarkan statistik dataset asli
    data = np.zeros((n_samples, 8))
    
    # Pregnancies (0-17)
    data[:, 0] = np.random.randint(0, 18, n_samples)
    
    # Glucose (0-199)
    data[:, 1] = np.random.normal(120, 30, n_samples).clip(0, 199)
    
    # BloodPressure (0-122)
    data[:, 2] = np.random.normal(70, 12, n_samples).clip(0, 122)
    
    # SkinThickness (0-99)
    data[:, 3] = np.random.normal(20, 10, n_samples).clip(0, 99)
    
    # Insulin (0-846)
    data[:, 4] = np.random.exponential(80, n_samples).clip(0, 846)
    
    # BMI (0-67)
    data[:, 5] = np.random.normal(32, 8, n_samples).clip(0, 67)
    
    # DiabetesPedigreeFunction (0.08-2.42)
    data[:, 6] = np.random.exponential(0.5, n_samples).clip(0.08, 2.42)
    
    # Age (21-81)
    data[:, 7] = np.random.randint(21, 82, n_samples)
    
    # Create target (35% diabetes rate as in original)
    # Simple rule: high glucose + high BMI + age > 50 increases risk
    risk_score = (
        (data[:, 1] > 140) * 0.4 +           # High glucose
        (data[:, 5] > 30) * 0.3 +            # High BMI
        (data[:, 7] > 50) * 0.2 +            # Age > 50
        (data[:, 0] > 5) * 0.1               # Many pregnancies
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.1, n_samples)
    
    # 35% positive rate
    y = (risk_score > np.percentile(risk_score, 65)).astype(int)
    
    print(f"Created dataset: {n_samples} samples")
    print(f"Diabetes rate: {y.mean():.1%}")
    
    return data, y

def train_model():
    print("Training diabetes prediction model...")
    
    # Create data
    X, y = create_diabetes_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Save model
    joblib.dump(model, "model.pkl")
    
    # Save feature names
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    joblib.dump(feature_names, "features.pkl")
    
    print("Model saved as model.pkl")
    
    return model

if __name__ == "__main__":
    train_model()