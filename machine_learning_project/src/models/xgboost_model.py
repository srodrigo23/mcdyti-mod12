from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.replace('PrivacySuppressed', 0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.mean())
    return df

def train_xgboost_classifier(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def train_xgboost_regressor(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def main():
    filepath = 'data/Most-Recent-Cohorts-Scorecard-Elements.csv'
    df = load_data(filepath)
    df = preprocess_data(df)

    # Example for classification
    X_class = df.drop('repayment_success', axis=1)  # Replace with actual target column
    y_class = df['repayment_success']  # Replace with actual target column
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    xgboost_classifier = train_xgboost_classifier(X_train_class, y_train_class)
    accuracy, report = evaluate_classifier(xgboost_classifier, X_test_class, y_test_class)
    print(f"Accuracy: {accuracy}")
    print(report)

    # Example for regression
    X_reg = df.drop('completion_rate', axis=1)  # Replace with actual target column
    y_reg = df['completion_rate']  # Replace with actual target column
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    xgboost_regressor = train_xgboost_regressor(X_train_reg, y_train_reg)
    mse = evaluate_regressor(xgboost_regressor, X_test_reg, y_test_reg)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()