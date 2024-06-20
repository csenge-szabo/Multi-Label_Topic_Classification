import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords

def undersample_data(input_file, output_file):
    """
    Reduces the frequency of overrepresented subtopics to the average count across all subtopics by random undersampling.
    
    Parameters:
    - input_file (str): Path to the CSV file containing the original data.
    - output_file (str): Path to save the balanced CSV file.
    """
    subtopics = [
    'Waiting time', 'Speaking to the right person', 'Correctness of handling', 'Functionalities web & app', 'Ease of process',
    'Reception & Registration', 'Friendliness', 'Quality of information', 'Information provision web & app', 'Clarity of information', 'Solution oriented',
    'Availability of employee', 'Price & costs', 'Speed of processing', 'Professionalism', 'Opening hours & accessibility', 'Ease of use web & app', 
    'Keeping up to date', 'Integrity & fulfilling responsibilities', 'Payout & return', 'No subtopic found', 'Quality of customer service', 'Facilities', 
    'Objection & evidence', 'General experience subtopic', 'Efficiency of process', 'Genuine interest', 'Expertise', 'Helpfulness', 'Personal approach', 
    'Communication'
    ]
    protected_subtopics = [
    'Speaking to the right person', 'Correctness of handling', 'Functionalities web & app', 
    'Reception & Registration', 'Quality of information', 'Information provision web & app', 
    'Availability of employee', 'Price & costs', 'Professionalism', 'Opening hours & accessibility', 
    'Ease of use web & app', 'Keeping up to date', 'Integrity & fulfilling responsibilities', 
    'Payout & return', 'Quality of customer service', 'Facilities', 'Objection & evidence', 
    'Efficiency of process', 'Genuine interest', 'Expertise', 'Personal approach', 'Communication'
    ]
    df = pd.read_csv(input_file, delimiter=';', header=0)
    # Setting a seed for reproducibility
    np.random.seed(12)

    # Calculate the occurrences of each subtopic to get average count
    subtopics_counts = df[subtopics].sum().sort_values(ascending=False)
    average_count = subtopics_counts.mean() 
    print(f"Average count of subtopic labels in {input_file}: {int(average_count)}")

    # Identify overrepresented subtopics
    overrepresented_subtopics = subtopics_counts[subtopics_counts > average_count]
    print(f"\nThe overrepresented topics are: {overrepresented_subtopics}")

    for subtopic in overrepresented_subtopics.index:
        current_count = df[subtopic].sum()
        while current_count > average_count:
            # Try to remove instances where the subtopic is the only label
            only_subtopic_mask = (df[subtopics].sum(axis=1) == 1) & (df[subtopic] == 1)
            only_subtopic_indices = df[only_subtopic_mask].index
            np.random.shuffle(only_subtopic_indices.values)  
            excess = int(current_count - average_count)
            n_remove = min(len(only_subtopic_indices), excess)
            if n_remove > 0:
                df = df.drop(only_subtopic_indices[:n_remove])
                
            else:
                # If no instances left where only the subtopic is rpesent, remove from any other "mixed" instances (containing other subtopics)
                # while preserving the protected subtopics that are underrepresented
                mixed_subtopic_indices = df[(df[subtopic] == 1) & (~df[protected_subtopics].any(axis=1))].index
                np.random.shuffle(mixed_subtopic_indices.values)
                n_remove = min(len(mixed_subtopic_indices), excess)
                if n_remove > 0:
                    df = df.drop(mixed_subtopic_indices[:n_remove])
                else:
                    break

            current_count = df[subtopic].sum()

    # Saving the balanced DataFrame to a new CSV file
    df.to_csv(output_file, index=False, sep=';')
    print(f"\nBalanced data saved to {output_file}.")

    # Print new subtopic counts for verification
    new_subtopics_counts = df[subtopics].sum()
    print("\nNew Subtopic Frequencies after data balancing:")
    for subtopic in subtopics:
        print(f"{subtopic}: {new_subtopics_counts[subtopic]}")

def preprocess_synthetic_data(input_file, output_file, lowercase=True, remove_stopwords=True, remove_punctuation=True, remove_digits=True, lemmatize=False):
    """
    Preprocesses synthetic data, adjusting text data by applying operations like lowercasing, stopword removal, punctuation removal, digit removal, and optional lemmatization.
    Parameters:
    - input_file (str): Path to the CSV file containing the original data.
    - output_file (str): Path to save the preprocessed CSV file.
    - lowercase (bool): If True, converts all text to lowercase. Default is True.
    - remove_stopwords (bool): If True, removes all stopwords as defined by the NLTK library. Default is True.
    - remove_punctuation (bool): If True, removes all punctuation characters. Default is True.
    - remove_digits (bool): If True, removes all digit characters. Default is True.
    - lemmatize (bool): If True, applies lemmatization to each token using Spacy. Default is False.
    Returns: None
    """
    df = pd.read_csv(input_file, delimiter=';')
    nlp = spacy.load('en_core_web_sm')
    
    # Function to preprocess a single text instance
    def preprocess_document(doc):
        doc = nlp(doc)
        tokens = [token.text for token in doc]

        if lowercase:
          tokens = [token.lower() for token in tokens]

        if remove_stopwords:
          stop_words = set(stopwords.words('english'))
          other_words = {"n't", "'s", "'m", "'ve", "'ll", "'re"}
          stop_words.update(other_words)
          tokens = [token for token in tokens if token.lower() not in stop_words]

        if remove_punctuation:
          punctuation_list = ['.', ',', '!', '?', '/', '(', ')', '@', '&', '*', ':', "'", '"', '...', '..', ';', '%', '=', '+', '.....', '....', "``", "''"]
          tokens = [token for token in tokens if token not in punctuation_list]

        if remove_digits:
          tokens = [token for token in tokens if not token.isdigit()]

        if lemmatize:
          tokens = [token.lemma_ for token in nlp(' '.join(tokens))]

        return ' '.join(tokens)

    # Filter the dataframe to process only rows with IDs starting with '1', these are synthetic data instances
    df_filtered = df[df['id'].astype(str).str.startswith('1')]
    df.loc[df_filtered.index, 'text'] = df_filtered['text'].apply(preprocess_document)

    df.to_csv(output_file, sep=';', index=False)
    print(f'Preprocessed file saved to {output_file}.')

def main():
    print("Undersample Data")
    input_file_undersample = input("Enter the path to the input CSV file for undersampling: ") # '../data/train.tsv'
    output_file_undersample = input("Enter the path to the output CSV file for undersampling: ") # '../data/undersampled_train.tsv'
    undersample_data(input_file_undersample, output_file_undersample)

    print("\nPreprocess Synthetic Data")
    input_file_preprocess = input("Enter the path to the input CSV file for preprocessing: ") # '../data/undersampled_train.tsv'
    output_file_preprocess = input("Enter the path to the output CSV file for preprocessing: ") # '../data/undersampled_train_2.tsv'
    preprocess_synthetic_data(input_file_preprocess, output_file_preprocess)

if __name__ == "__main__":
    nltk.download('stopwords')
    main()