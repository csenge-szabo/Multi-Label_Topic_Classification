import pandas as pd

def analyze_confusion(focus_topic, gold_data, pred_data, topics):
    """
    Analyzes the confusion between predicted and actual topic labels, focusing on a specified topic.
    
    Parameters:
        focus_topic (str): The topic to focus the analysis on.
        gold_data (DataFrame): The DataFrame containing the gold standard labels.
        pred_data (DataFrame): The DataFrame containing the predicted labels.
        subtopics (list): List of all possible subtopics.
    """
    if focus_topic not in topics:
        print(f"Topic '{focus_topic}' not found in the topics list.")
        return

    cols = ['id', 'text'] + topics
    gold = gold_data[cols]
    predicted = pred_data[cols]
    
    false_positives = predicted[(predicted[focus_topic] == 1) & (gold[focus_topic] == 0)]
    false_negatives = predicted[(predicted[focus_topic] == 0) & (gold[focus_topic] == 1)]
    
    print(f"Analysis for topic: {focus_topic}")
    print(f"Number of False Positives: {len(false_positives)}")
    print(f"Number of False Negatives: {len(false_negatives)}\n")
    
    for index, row in false_positives.iterrows():
        print(f"False Positive: {row['text']}")
    
    for index, row in false_negatives.iterrows():
        print(f"False Negative: {row['text']}")

    confusion_details = {}
    for topic in topics:
        if topic != focus_topic:
            confused_cases = predicted[(predicted[focus_topic] == 0) & (predicted[topic] == 1) & (gold[focus_topic] == 1)]
            confusion_details[topic] = {
                'count': confused_cases.shape[0],
                'instances': confused_cases['text'].tolist()}  # Collecting text instances
            
    confused_topics = sorted(confusion_details.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"Topics most frequently confused with '{focus_topic}':")
    for topic, details in confused_topics:
        if details['count'] > 0:
            print(f"\n{topic}: {details['count']} times")
            for detail in details['instances']:
                print(f"{detail}")

def lookup_instance(instance_id, gold_data, pred_data, subtopics):
    """
    Looks up the labels for a specific instance by its ID and compares the gold labels with the predictions.
    
    Parameters:
        instance_id (int): The ID of the instance to look up.
        gold_data (DataFrame): The DataFrame containing the gold standard labels.
        pred_data (DataFrame): The DataFrame containing the predicted labels.
        subtopics (list): List of all possible subtopics.
    """
    gold_row = gold_data[gold_data['id'] == instance_id]
    pred_row = pred_data[pred_data['id'] == instance_id]

    if gold_row.empty or pred_row.empty:
        print("No instance found with the specified ID.")
        return
    
    print(f"{instance_id}")
    print(f"{gold_row.iloc[0]['text']}\n")

    print("True Labels:")
    for topic in subtopics:
        if gold_row.iloc[0][topic] == 1:
            print(f" - {topic}")
    
    print("Predicted Labels:")
    for topic in subtopics:
        if pred_row.iloc[0][topic] == 1:
            print(f" - {topic}")

def main():
    # Load data
    gold_data = pd.read_csv('../data/test.csv', sep=';')
    pred_data = pd.read_csv('../model_predictions/bert_predictions_test_2step_oversampled.csv', sep=';') # or you can use '../model_predictions/bert_predictions_test_2step.csv' / '../model_predictions/bert_predictions_test_2step_undersampled.csv'
    gold_data.sort_values('id', inplace=True)
    pred_data.sort_values('id', inplace=True)
    assert all(gold_data['id'] == pred_data['id']), "IDs do not match between gold data and predictions."

    maintopics = ['Making contact with employee', 'Processes', 'Digital possibilities', 'General experience', 'Information provision', 'Employee attitude & behavior', 'Handling', 'No topic found', 'Knowledge & skills of employee', 'Price & quality', 'Physical service provision']
    subtopics = ['Waiting time', 'Speaking to the right person', 'Correctness of handling', 'Functionalities web & app', 'Ease of process', 'Reception & Registration', 'Friendliness', 'Quality of information', 'Information provision web & app', 'Clarity of information', 'Solution oriented', 'Availability of employee', 'Price & costs', 'Speed of processing', 'Professionalism', 'Opening hours & accessibility', 'Ease of use web & app', 'Keeping up to date', 'Integrity & fulfilling responsibilities', 'Payout & return', 'No subtopic found', 'Quality of customer service', 'Facilities', 'Objection & evidence', 'General experience subtopic', 'Efficiency of process', 'Genuine interest', 'Expertise', 'Helpfulness', 'Personal approach', 'Communication']
    all_topics = maintopics + subtopics

    # User input for analysis
    focus_topic = input("Enter the topic for confusion analysis: ")
    analyze_confusion(focus_topic, gold_data, pred_data, subtopics) # or replace subtopics with: maintopics/all_topics

    # User input for instance lookup
    instance_id = int(input("Enter the instance ID to lookup: "))
    lookup_instance(instance_id, gold_data, pred_data, subtopics)  # or replace subtopics with: maintopics/all_topics

if __name__ == "__main__":
    main()
