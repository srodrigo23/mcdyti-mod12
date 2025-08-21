# Machine Learning Project

This project implements various machine learning models, including Random Forest and XGBoost, along with segmentation and classification techniques. The goal is to analyze a dataset containing information about educational institutions and to evaluate the performance of different models using various metrics.

## Project Structure

- **data/**: Contains the dataset used for training and evaluating the machine learning models.
  - `Most-Recent-Cohorts-Scorecard-Elements.csv`: The dataset file.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis (EDA).
  - `eda.ipynb`: Notebook for EDA, including data visualization and initial insights.

- **src/**: Contains the source code for the project.
  - **models/**: Implements machine learning models.
    - `__init__.py`: Initializes the models package.
    - `random_forest.py`: Implementation of the Random Forest model.
    - `xgboost_model.py`: Implementation of the XGBoost model.
    - `segmentation.py`: Functions for segmentation analysis.
    - `classification.py`: Functions for classification tasks.
  
  - **utils/**: Contains utility functions.
    - `__init__.py`: Initializes the utils package.
    - `data_loader.py`: Functions for loading and preprocessing the dataset.
  
  - **evaluation/**: Contains evaluation metrics functions.
    - `__init__.py`: Initializes the evaluation package.
    - `metrics.py`: Functions for calculating evaluation metrics.
  
  - **visualization/**: Contains visualization functions.
    - `__init__.py`: Initializes the visualization package.
    - `plots.py`: Functions for creating visualizations.
  
  - `main.py`: Entry point for the project, orchestrating data loading, model training, evaluation, and visualization.
  
  - `requirements.txt`: Lists required Python packages for the project.

## Installation

To install the required packages, run the following command:

```
pip install -r src/requirements.txt
```

## Usage

1. Load the dataset using the functions in `src/utils/data_loader.py`.
2. Perform exploratory data analysis in the `notebooks/eda.ipynb`.
3. Train models using the implementations in `src/models/random_forest.py` and `src/models/xgboost_model.py`.
4. Evaluate model performance using functions in `src/evaluation/metrics.py`.
5. Visualize results using the functions in `src/visualization/plots.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.