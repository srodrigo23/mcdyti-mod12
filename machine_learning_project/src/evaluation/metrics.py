def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def recall(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

def evaluate_model(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy(y_true, y_pred),
        'Precision': precision(y_true, y_pred),
        'Recall': recall(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred)
    }
    return metrics