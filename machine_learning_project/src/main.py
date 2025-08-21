# Entry point for the machine learning project

import pandas as pd
from src.utils.data_loader import load_data
from src.models.random_forest import train_random_forest, evaluate_random_forest
from src.models.xgboost_model import train_xgboost, evaluate_xgboost
from src.models.segmentation import perform_segmentation
from src.models.classification import train_classification_model, evaluate_classification
from src.evaluation.metrics import calculate_metrics
from src.visualization.plots import plot_model_performance

def main():
    # Load the dataset
    data = load_data('data/Most-Recent-Cohorts-Scorecard-Elements.csv')

    # Perform segmentation analysis
    segmentation_results = perform_segmentation(data)

    # Train and evaluate Random Forest model
    rf_model, rf_predictions = train_random_forest(data)
    rf_metrics = evaluate_random_forest(data, rf_predictions)

    # Train and evaluate XGBoost model
    xgb_model, xgb_predictions = train_xgboost(data)
    xgb_metrics = evaluate_xgboost(data, xgb_predictions)

    # Train and evaluate classification model
    classification_model, classification_predictions = train_classification_model(data)
    classification_metrics = evaluate_classification(data, classification_predictions)

    # Store results in a DataFrame
    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Classification'],
        'Metrics': [rf_metrics, xgb_metrics, classification_metrics]
    })

    print(results_df)

    # Visualize model performance
    plot_model_performance(results_df)

if __name__ == "__main__":
    main()