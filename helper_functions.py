import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def plot_metrics(model_history):
    """
        Function to Plot the graphs of Accuracy and Losses wrt Number of Epochs
    """
    # Epochs vs Accuracy
    plt.title('Epochs vs Accuracy')
    plt.plot(model_history.history['accuracy'], color='red', lable='Training')
    plt.plot(model_history.history['val_accuracy'], color='blue', lable='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'])

    # Epochs vs Loss
    plt.title('Epochs vs Loss')
    plt.plot(model_history.history['loss'], color='red', lable='Training')
    plt.plot(model_history.history['val_loss'], color='blue', lable='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'])


def get_metrics(y, y_pred):
    """
        Fuction to Evaluate Model Performance
    """
    accuracy = accuracy_score(y_pred, y)
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)
    f1 = f1_score(y_pred, y)
    cr = classification_report(y_pred, y)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Classification Report\n": cr
    }

    return metrics


def assign_weights(X, y):
    """
        Assign Weights to account for class imbalance
    """
    classes = np.unique(y)
    target_weights = class_weight.compute_class_weight('balanced', classes, y.ravel())
    target_weights = {target:target_weights[target] for target in range(len(target_weights))}
    return target_weights
