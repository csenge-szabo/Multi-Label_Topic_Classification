import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import classification_report, hamming_loss
from SVMs_1step import vectorize_texts, train_and_predict, write_predictions, plot_confusion_matrices

def load_and_split_data(filename):
    """
    Loads data from a specified CSV file and splits it into main topics and subtopics.
    Parameters:
    - filename (str): The path to the CSV file containing the dataset.
    Returns:
    - tuple: Contains IDs, texts, main topic labels, sub topic labels, and names of main and sub topic labels.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader)

        main_label_names = header[2:13]  # Main topic labels
        sub_label_names = header[13:]  # Subtopic labels

        ids, texts, main_labels, sub_labels = [], [], [], []
        
        for row in reader:
            ids.append(row[0])
            texts.append(row[1])
            main_labels.append(np.array([int(x) for x in row[2:13]]))
            sub_labels.append(np.array([int(x) for x in row[13:]]))

        return ids, texts, np.array(main_labels), np.array(sub_labels), main_label_names, sub_label_names

def global_hyperparameter_tuning(X_train, y_train_main, y_train_sub, param_file):
    """
    Performs global hyperparameter tuning for both main and subtopics.
    Parameters:
    - X_train (sparse matrix): Training data in TF-IDF format.
    - y_train_main (array): Main topic training labels.
    - y_train_sub (array): Subtopic training labels.
    - param_file (str): Path to the file where the best hyper-parameters will be stored.
    """
    if not os.path.exists(param_file):

        parameters = {
            'classifier': [LinearSVC(random_state=0)],
            'classifier__C': [0.01, 0.1, 1.0],
            'classifier__tol': [1e-4, 1e-3, 1e-2],
            'classifier__loss': ['hinge', 'squared_hinge']
        }

        best_overall_score = 0
        best_parameters = None

        # tune parameters for main topics
        grid_search_main = GridSearchCV(BinaryRelevance(), param_grid=parameters, cv=10, scoring='f1_macro', n_jobs=1, verbose=1)
        grid_search_main.fit(X_train, y_train_main)
        main_score = grid_search_main.best_score_
        main_params = grid_search_main.best_params_

        # tune parameters for subtopics
        grid_search_sub = GridSearchCV(BinaryRelevance(), param_grid=parameters, cv=10, scoring='f1_macro', n_jobs=1, verbose=1)
        grid_search_sub.fit(X_train, y_train_sub)
        sub_score = grid_search_sub.best_score_
        sub_params = grid_search_sub.best_params_

        overall_score = (main_score + sub_score) / 2
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_parameters = main_params if main_score > sub_score else sub_params
            print(f"Best parameters: {best_parameters} with overall score: {overall_score}")
        
            # Save the best parameters to a file
            with open(param_file, 'wb') as f:
                pickle.dump(best_parameters, f)
    else:
        print("Optimized parameters already exist.")

def subtopic_prediction(maintopics, subtopics, label_map, train_file, main_topic_predictions_file, final_output, param_file):
    """
    Executes subtopic classification based on the predictions of main topics in the first step.
    Parameters:
    - maintopics (list of str): List of main topic labels.
    - subtopics (list of str): List of subtopic labels.
    - label_map (dict): Dictionary mapping main topics to corresponding subtopics.
    - train_file (str): Path to the training dataset file.
    - main_topic_predictions_file (str): Path to the file containing predictions for main topics.
    - final_output (str): Path to the output file where the final predictions will be stored.
    - param_file (str): Path to the global hyperparameters file.
    Returns: None
    Outputs the final predictions for both main and subtopics in a single file.
    """
    # Load full training and predicted data
    full_train_data = pd.read_csv(train_file, delimiter=';')
    full_pred_data = pd.read_csv(main_topic_predictions_file, delimiter=';')

    # Initialize predictions DataFrame with default values (0 for all subtopics)
    final_predictions = full_pred_data[['id', 'text'] + maintopics].copy()  # include main topic labels
    for subtopic in subtopics:
        final_predictions[subtopic] = 0

    # Iterate over each main topic
    for main_topic in maintopics:
        subtopic_labels = label_map[main_topic]
        print(f"Processing for main topic: {main_topic}")
        print(f'Subtopic labels found: {subtopic_labels}\n')

        if len(subtopic_labels) == 1: # In case of "No topic found" and "General experience", there is only 1 subtopic, so we assign the same value as predicted for the main topic
            single_subtopic = subtopic_labels[0]
            print(f"Direct assignment for {main_topic} as it has only one subtopic: {single_subtopic}\n")
            final_predictions[subtopic_labels[0]] = final_predictions[main_topic]
            continue

        # Filter training data and the predicted data from step 1 for the current main topic
        train_data = full_train_data[full_train_data[main_topic] == 1].copy()
        pred_data = full_pred_data[full_pred_data[main_topic] == 1].copy()

        # Ensure all text entries are strings and replace NaNs. Due to preprocessing, if the text only contained stopwords for example, they were removed.
        train_data['text'] = train_data['text'].fillna('')
        pred_data['text'] = pred_data['text'].fillna('')

        # Extract texts and labels
        X_train_texts = train_data['text']
        y_train = train_data[subtopic_labels].values
        X_test_texts = pred_data['text']

        X_train, X_test = vectorize_texts(X_train_texts, X_test_texts)

        y_pred = train_and_predict(X_train, y_train, X_test, param_file, main_topic)
        y_pred = y_pred.toarray()
  
        # Update predictions in the final DataFrame
        for idx, subtopic in enumerate(subtopic_labels):
            final_predictions.loc[final_predictions['id'].isin(pred_data['id']), subtopic] = y_pred[:, idx]

    # Write all predictions to a single file
    final_predictions.to_csv(final_output, index=False, sep=';')
    print(f"All predictions written to {final_output}")

def main():
    maintopics = ['Making contact with employee', 'Processes', 'Digital possibilities', 'General experience', 'Information provision', 'Employee attitude & behavior', 'Handling', 'No topic found', 'Knowledge & skills of employee', 'Price & quality', 'Physical service provision']
    subtopics = ['Waiting time', 'Speaking to the right person', 'Correctness of handling', 'Functionalities web & app', 'Ease of process', 'Reception & Registration', 'Friendliness', 'Quality of information', 'Information provision web & app', 'Clarity of information', 'Solution oriented', 'Availability of employee', 'Price & costs', 'Speed of processing', 'Professionalism', 'Opening hours & accessibility', 'Ease of use web & app', 'Keeping up to date', 'Integrity & fulfilling responsibilities', 'Payout & return', 'No subtopic found', 'Quality of customer service', 'Facilities', 'Objection & evidence', 'General experience subtopic', 'Efficiency of process', 'Genuine interest', 'Expertise', 'Helpfulness', 'Personal approach', 'Communication']
    label_map = {
    'Making contact with employee': ['Waiting time', 'Availability of employee', 'Speaking to the right person'], 
    'Processes': ['Ease of process', 'Efficiency of process'], 
    'Digital possibilities': [ 'Functionalities web & app', 'Information provision web & app', 'Ease of use web & app'], 
    'General experience': ['General experience subtopic'], 
    'Information provision': ['Clarity of information', 'Quality of information','Communication',  'Integrity & fulfilling responsibilities', 'Keeping up to date'], 
    'Employee attitude & behavior': ['Friendliness','Helpfulness', 'Personal approach','Genuine interest'], 
    'Handling': ['Speed of processing','Correctness of handling','Objection & evidence'], 
    'No topic found': ['No subtopic found'], 
    'Knowledge & skills of employee': ['Solution oriented','Expertise', 'Quality of customer service', 'Professionalism'], 
    'Price & quality': ['Price & costs', 'Payout & return'], 
    'Physical service provision': ['Reception & Registration',  'Opening hours & accessibility', 'Facilities']
    }
    
    train_file = '../data/train_2.csv'
    valid_file = '../data/validation_2.csv'
    test_file = '../data/test_2.csv'
    undersampled_train_file = '../data/undersampled_train_2.csv'
    oversampled_train_file = '../data/oversampled_train_2.csv'
    param_file = '../hyperparameters/best_params_2step.pkl'
    main_topic_predictions_file = '../model_predictions/svms_first_step.csv'
    two_step_final_predictions_file = '../model_predictions/svms_predictions_2step.csv' # or '../model_predictions/svms_predictions_2step_oversampled.csv' / '../model_predictions/svms_predictions_2step_undersampled.csv'
    
    # Load data
    train_ids, train_texts, y_train_main, y_train_sub, main_label_classes, sub_label_classes = load_and_split_data(train_file)
    test_ids, test_texts, y_test_main, y_test_sub, _, _ = load_and_split_data(test_file)
    
    # Vectorize texts
    X_train, X_test = vectorize_texts(train_texts, test_texts)

    # Hyper-parameter tuning
    global_hyperparameter_tuning(X_train, y_train_main, y_train_sub, param_file)

    # First classification step - model predicts Main Topics
    Y_pred_main = train_and_predict(X_train, y_train_main, X_test, param_file, 'Main Topics')
    write_predictions(main_topic_predictions_file, main_label_classes, test_ids, test_texts, Y_pred_main)
    print("Predictions for main topics finished.")

    # Second classification step - model predicts Main Topics
    subtopic_prediction(maintopics, subtopics, label_map, train_file, main_topic_predictions_file, two_step_final_predictions_file, param_file)
    print("Predictions for subtopics finished.")

    # Evaluation
    predicted_data = pd.read_csv(two_step_final_predictions_file, delimiter=';')
    true_data = pd.read_csv(test_file, delimiter=';')
    # Sort the data by 'id' to align rows
    predicted_data = predicted_data.sort_values(by='id').reset_index(drop=True)
    true_data = true_data.sort_values(by='id').reset_index(drop=True)
    # Ensure that both DataFrames have the same row order
    assert all(predicted_data['id'] == true_data['id']), "IDs do not match."
    # Columns to be used for generating the classification report
    label_columns = [col for col in predicted_data.columns if col not in ['id', 'text']]
    report_dict = classification_report(true_data[label_columns], predicted_data[label_columns], target_names=label_columns, zero_division=0, output_dict=True)
    print(classification_report(true_data[label_columns], predicted_data[label_columns], target_names=label_columns, zero_division=0))
    hamming_loss_value = "{:.3f}".format(hamming_loss(true_data[label_columns], predicted_data[label_columns]))
    print("Hamming Loss:", hamming_loss_value)

    # Plot confusion matrices for each label
    y_true = true_data[label_columns].to_numpy()
    y_pred = predicted_data[label_columns].to_numpy()
    plot_confusion_matrices(y_true, y_pred, label_columns)

    # Save evaluation results
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)
    report_df.to_csv('../results/SVMs/test_report_2step.csv', sep=';', index=True) # or replace with '../results/SVMs/test_report_2step_undersampled.csv' or '../results/SVMs/test_report_2step_oversampled.csv'
    print("Classification report saved to file.")

if __name__ == "__main__":
    main()