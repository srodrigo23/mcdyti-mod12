from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Example preprocessing: drop rows with missing target values
    df = df.dropna(subset=['target_column'])  # Replace 'target_column' with the actual target column name
    X = df.drop('target_column', axis=1)  # Features
    y = df['target_column']  # Target variable
    return X, y

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

def main(filepath):
    df = load_data(filepath)
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_random_forest(X_train, y_train)
    
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    results = pd.DataFrame({
        'Metric': ['Accuracy'],
        'Score': [accuracy]
    })
    
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    return results

if __name__ == "__main__":
    filepath = '../data/Most-Recent-Cohorts-Scorecard-Elements.csv'  # Adjust path as necessary
    results_df = main(filepath)