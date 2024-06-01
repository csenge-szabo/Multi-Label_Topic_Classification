import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import pickle

def load_data(filename):
    """
    Load texts, labels, and label names from a CSV file.
    Parameters:
    - filename (str): Path to the CSV file.
    Returns:
    - tuple: IDs, texts, labels, and label names.
    """
    ids, texts, labels = [], [], []
    label_names = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')

        header = next(reader)  # Get the header to extract label names
        label_names = header[2:]  # label names start from the third column
        
        for row in reader:
            ids.append(row[0])
            texts.append(row[1])  # text is in the second column
            labels.append(np.array([int(label) for label in row[2:]]))  # Convert labels to integers and use numpy array
    
    return ids, texts, np.array(labels), label_names

def vectorize_texts(train_texts, test_texts):
    """
    Converts text data into TF-IDF features.
    Parameters:
    - train_texts: List of training text instances.
    - test_texts: List of test text instances.
    Returns:
    - tuple: TF-IDF vectorized training and test data.
    inspired by Piek Vossen, [https://github.com/cltl/ma-hlt-labs/blob/master/lab3.machine_learning/Lab3.5.ml.emotion-detection-bow.ipynb]
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf

def hyperparameter_tuning(X_train, y_train, param_file):
    """
    Performs hyper-parameter tuning using GridSearchCV and saves the best parameters to a file.
    Parameters:
    - X_train (sparse matrix): Training data in TF-IDF format.
    - y_train (array): Training labels.
    - param_file (str): Path to the file where best hyper-parameters will be saved.
    Returns: None
    Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    if not os.path.exists(param_file):
        parameters = {
            'classifier': [LinearSVC(random_state=0)],
            'classifier__C': [0.01, 0.1, 1.0],
            'classifier__tol': [1e-4, 1e-3, 1e-2],
            'classifier__loss': ['hinge', 'squared_hinge']
        }

        grid_search_main = GridSearchCV(BinaryRelevance(), param_grid=parameters, cv=10, scoring='f1_macro', n_jobs=1, verbose=1)
        grid_search_main.fit(X_train, y_train)
        score = grid_search_main.best_score_
        best_parameters = grid_search_main.best_params_

        print(f"Best parameters: {best_parameters} with overall score: {score}")
        
        # Save the best parameters to a file
        with open(param_file, 'wb') as f:
            pickle.dump(best_parameters, f)
    else:
        print("Optimized parameters already exist.")

def train_and_predict(X_train, y_train, X_test, param_file, step_label):
    """
    Trains a classifier with binary relevance and LinearSVC, performs predictions. It also includes hyper-parameter optimization.
    
    Parameters:
    - X_train (sparse matrix): Training data in TF-IDF format.
    - y_train (array): Training labels.
    - X_test (sparse matrix): Test data in TF-IDF format.
    - param_file (str): Path to pickle file storing the best hyper-parameters.
    - step_label (str): String to define the type of classification (1-step or 2-step).
    
    Returns:
    - array: Predicted labels.
    """
    # Load the best parameters
    with open(param_file, 'rb') as f:
        best_params = pickle.load(f)
    print(f'Using loaded parameters for {step_label}:', best_params)

    # Extract only the relevant parameters for LinearSVC
    svc_params = {k.replace('classifier__', ''): v for k, v in best_params.items() if 'classifier__' in k}
    classifier = BinaryRelevance(classifier=LinearSVC(**svc_params))
    
    # Train and predict
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def write_predictions(filename, label_classes, test_ids, test_texts, predicted_labels):
    """
    Writes predictions to a CSV file along with text IDs and texts.
    Parameters:
    - filename (str): Path to the output CSV file.
    - label_classes (list): List of label class names.
    - test_ids (list): List of test text IDs.
    - test_texts (list): List of test texts.
    - predicted_labels (array): Predicted labels.
    Returns: None
    """
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['id','text'] + label_classes) 
        for id, text, pred in zip(test_ids, test_texts, predicted_labels):
            writer.writerow([id, text] + list(map(str, pred.toarray()[0])))

def plot_confusion_matrices(y_true, y_pred, classes):
    """
    Plots confusion matrices for each label class separately.
    """
    for i, class_name in enumerate(classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(class_name)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

def main():
    train_file = '../data/train_2.csv'
    valid_file = '../data/validation_2.csv'
    test_file = '../data/test_2.csv'
    output_file = '../model_predictions/svms_predictions_1step.csv'
    param_file = '../hyperparameters/best_params_1step.pkl'

    # Load datasets
    _, train_texts, y_train, label_classes = load_data(train_file)
    test_ids, test_texts, y_test, _ = load_data(test_file)

    # Vectorize texts
    X_train, X_test = vectorize_texts(train_texts, test_texts)

    # Hyper-parameter tuning
    hyperparameter_tuning(X_train, y_train, param_file)

    # Train and predict with hyperparameter tuning
    Y_pred = train_and_predict(X_train, y_train, X_test, param_file, '1-step classification')
    write_predictions(output_file, label_classes, test_ids, test_texts, Y_pred)
    
    # Evaluation
    report_dict = classification_report(y_test, Y_pred.toarray(), target_names=label_classes, zero_division=0, output_dict=True)
    print(classification_report(y_test, Y_pred.toarray(), target_names=label_classes, zero_division=0))
    hamming_loss_value = "{:.3f}".format(hamming_loss(y_test, Y_pred.toarray()))
    print("Hamming Loss:", hamming_loss_value)
    plot_confusion_matrices(y_test, Y_pred.toarray(), label_classes)

    # Save evaluation results
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)
    report_df.to_csv('../results/SVMs/test_report_1step.csv', sep=';', index=True)
    print("Classification report saved to file.")

if __name__ == "__main__":
    main()
