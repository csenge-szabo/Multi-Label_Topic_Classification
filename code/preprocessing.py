import spacy
import re
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller

def preprocess_text(text, nlp, lowercase=False, spell_check=False, remove_stopwords=False, remove_punctuation=False, remove_digits=False, lemmatize=False):
    """
    Preprocesses text data based on specified options such as tokenization, spell-check, stopword removal, 
    punctuation removal, digit removal, and lemmatization. It also masks privacy-sensitive information.
    
    Parameters:
    - text (str): Input text to preprocess.
    - nlp (Language): Spacy NLP model object.
    - lowercase (bool): Flag to lowercase the text.
    - spell_check (bool): Flag to perform spell check.
    - remove_stopwords (bool): Flag to remove stopwords.
    - remove_punctuation (bool): Flag to remove punctuation.
    - remove_digits (bool): Flag to remove digits.
    - lemmatize (bool): Flag to lemmatize the text.
    
    Returns:
    - str: Preprocessed text.
    """
    # Named Entity recognition and masking privacy-sensitive information 
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, 'PERSON')
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            text = text.replace(ent.text, 'LOCATION')

    # Define parameters for masking
    email_pattern = r'[A-Za-z0-9]+[.-_]?[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    url_pattern = r'\b(?:https?:\/\/)?[\w\-_]+(\.[\w\-_]+)+(\.nl)[\/?][\w\-_.,@?^=%&:\/~+#]*\b|\b[\w\-_.]+@[\w\-_]+(\.[\w\-_]+)+(\.nl)\b' 
    date_pattern = r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s\d{1,2}(?:st|nd|rd|th)?)?\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{2,4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    time_pattern = r'(?<!€)\b(?:[01]?[0-9]|2[0-3])(?::[0-5][0-9]|.[0-5][0-9]|(?:(?:\s?o\'?clock)))(?:\s?(?:AM|PM|am|pm|a.m.|p.m.))?[.,]?\b(?!€)'

    # Compile regex patterns
    email_regex = re.compile(email_pattern)
    url_regex = re.compile(url_pattern)
    date_regex = re.compile(date_pattern)
    time_regex = re.compile(time_pattern)
    
    # Apply masking
    text = email_regex.sub('EMAIL', text)
    text = url_regex.sub('URL', text)
    text = date_regex.sub('DATE', text)
    text = time_regex.sub('TIME', text)
    text = text.replace('Rotterdam', 'LOCATION').replace('rotterdam', 'LOCATION').replace('ROTTERDAM', 'LOCATION')

    # If you only want to use the function for privacy masking, uncomment the line below and comment out the rest of the function
    # return text

    # Text preprocessing steps
    tokens = word_tokenize(text)
    if lowercase:
        # Tokenization and lowercasing
        tokens = [token.lower() for token in tokens]
        
    # Correcting spelling
    if spell_check:
        spell = Speller(lang='en')
        tokens = [spell(token) for token in tokens]
        
    # Removal of stopwords and other common words
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Remove other commonly occurring tokens and tokens with encoding problems
        other_words = {"n't", "'s", "'m", "'ve", "'ll", "'re", '==&gt;', '&amp', 'a&amp;b', 'q&amp;a', 'mr', 'mrs', 'blabla009', 'blabla0010', 'Mrs.', 'Mr.'}
        stop_words.update(other_words)
        tokens = [token for token in tokens if token.lower() not in stop_words]
        
    # Removal of punctuation marks
    if remove_punctuation:
        punctuation_list = ['.', ',', '!', '?', '/', '(', ')', '@', '&', '*', ':', "'", '"', '...', '..', ';', '%', '=', '+', '.....', '....', "``", "''"]
        tokens = [token for token in tokens if token not in punctuation_list]

    # Removal of digits
    if remove_digits:
            tokens = [token for token in tokens if not token.isdigit()]

    # Lemmatization
    if lemmatize:
        tokens = [token.lemma_ for token in nlp(' '.join(tokens))]

    return ' '.join(tokens)

def split_dataset(processed_df):
    """
    Splits the dataset into training, validation, and test sets (80-10-10 ratio), and writes them to CSV files.
    
    Parameters:
    - processed_df (DataFrame): The dataframe containing the preprocessed text and labels.
    Returns: None

    Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    """
    texts = processed_df['text']
    labels = processed_df.drop(['id', 'text'], axis=1)

    indices = np.arange(len(texts))
    msss_initial = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index_temp in msss_initial.split(indices, labels):
        break

    dev_index, test_index = train_test_split(test_index_temp, test_size=0.5, random_state=42)
    datasets = {'data/train.csv': train_index, 
                'data/validation.csv': dev_index, 
                'data/test.csv': test_index
                }

    for filename, indices in datasets.items():
        processed_df.iloc[indices].to_csv(filename, sep=';', index=False, encoding='utf-8')        
        print(f"Dataset written to {filename}")

def load_and_process_data(input_file, nlp, all_labels):
    """
    Loads data from a TSV file, processes it using the specified NLP model, and converts labels into binary format (0 or 1).
    
    Parameters:
    - input_file (str): Path to the input TSV file.
    - nlp (Language): Spacy NLP model object.
    - all_labels (list): List of all possible labels.
    
    Returns:
    - DataFrame: The processed data with binary labels.
    """
    df = pd.read_csv(input_file, delimiter='\t', encoding='utf-8', header=None)
    
    df.columns = ['id', 'text'] + ['label{}'.format(i) for i in range(1, df.shape[1]-1)]
    df['text'] = df['text'].apply(lambda text: preprocess_text(text, nlp, lowercase=True, spell_check=False, remove_stopwords=True, remove_punctuation=False, remove_digits=False, lemmatize=False))

    # Handling multiple labels per instance
    label_data = df.iloc[:, 2:].apply(lambda x: x.dropna().tolist(), axis=1)
    mlb = MultiLabelBinarizer(classes=all_labels)
    binary_labels = mlb.fit_transform(label_data)
    label_frame = pd.DataFrame(binary_labels, columns=mlb.classes_)

    # Concatenate id, text, and new binary labels
    processed_df = pd.concat([df[['id', 'text']], label_frame], axis=1)
    return processed_df

def write_data(output_file, processed_df):
    """
    Writes the processed data to a CSV file (used for the full dataset).
    
    Parameters:
    - output_file (str): Path to the output file.
    - processed_df (DataFrame): The dataframe to write.
    Returns: None
    """
    label_cols = processed_df.columns[processed_df.columns.get_loc('text')+1:]
    
    # Convert all label columns to integer type
    processed_df[label_cols] = processed_df[label_cols].astype(int)
    processed_df.to_csv(output_file, index=False, sep=';', encoding='utf-8')

def main():
    nltk.download('stopwords')
    nlp = spacy.load('en_core_web_sm')

    input_file = input('Enter the path to the input file: ') # '../data/converted_data.tsv'
    output_file = input('Enter the path to the output file: ') # '../data/full_dataset.csv'

    maintopics = ['Making contact with employee', 'Processes', 'Digital possibilities', 'General experience', 'Information provision', 'Employee attitude & behavior', 'Handling', 'No topic found', 'Knowledge & skills of employee', 'Price & quality', 'Physical service provision']
    subtopics = ['Waiting time', 'Speaking to the right person', 'Correctness of handling', 'Functionalities web & app', 'Ease of process', 'Reception & Registration', 'Friendliness', 'Quality of information', 'Information provision web & app', 'Clarity of information', 'Solution oriented', 'Availability of employee', 'Price & costs', 'Speed of processing', 'Professionalism', 'Opening hours & accessibility', 'Ease of use web & app', 'Keeping up to date', 'Integrity & fulfilling responsibilities', 'Payout & return', 'No subtopic found', 'Quality of customer service', 'Facilities', 'Objection & evidence', 'General experience subtopic', 'Efficiency of process', 'Genuine interest', 'Expertise', 'Helpfulness', 'Personal approach', 'Communication']
    all_labels = maintopics + subtopics

    processed_df = load_and_process_data(input_file, nlp, all_labels)
    print('Data loaded and preprocessed.')

    write_data(output_file, processed_df)
    print(f'Data is written to output file: {output_file}.')

    split_dataset(processed_df)
    print('Data splitting is done.')

if __name__ == "__main__":
    main()
