from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Drop missing values
    # Convert categorical variables to dummy variables if necessary
    df = pd.get_dummies(df)
    return df

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics

def main(filepath):
    df = load_data(filepath)
    df = preprocess_data(df)

    # Assuming 'target' is the name of the target variable
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [rf_metrics['accuracy'], xgb_metrics['accuracy']],
        'Precision': [rf_metrics['precision'], xgb_metrics['precision']],
        'Recall': [rf_metrics['recall'], xgb_metrics['recall']],
        'F1 Score': [rf_metrics['f1_score'], xgb_metrics['f1_score']]
    })

    return results_df

if __name__ == "__main__":
    filepath = '../data/Most-Recent-Cohorts-Scorecard-Elements.csv'
    results = main(filepath)
    print(results)