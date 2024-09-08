# %% [markdown]
# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Hong Thai Ngoc Ha, Dao Sy Trung Kien, Dao Quang Minh
# #### Student ID: S4060340, S3979613, S4015939
# 
# Date: 04/09/2024
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: 
# * pandas
# * re
# * numpy
# * os
# * nltk
# 
# ## Introduction
# This is the basic text preprocessing part of the project to get the brief information of job advertisements. This part focuses on preprocessing the 'Description' part of the collection of advertisements. To make sure the information is ready for the next steps, every required step for text preprocessing is absolutely performed clearly in this notebook.
# 

# %% [markdown]
# ## Importing libraries 

# %%
#Import all the necessary libaries
import re
import os
import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# %% [markdown]
# ### 1.1 Examining and loading data
# The dataset that is provided contains multiple folders with a collection of job advertisements. For each job advertisement, it is stored in the TXT file format.

# %%
# Create function to get the information of job advertisement
def get_job_infors(root_folder):
    job_infos = [] # Create an array to store all the information of advertisement
    # Get the id from filename
    id_pattern = re.compile(r'Job_(\d+)\.txt')
    target_subfolders = {'Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales'}
    # Go through all the file in each folder
    for foldername, subfolders, filenames in os.walk(root_folder):
        # Ensure to get all the data with the right folders
        current_folder = os.path.basename(foldername)
        if current_folder in target_subfolders:
            for filename in filenames: # For every filename in the filenames get from the folder
                if filename.endswith('.txt'): # If the filename contian '.txt', get that file
                    job_id_match = id_pattern.search(filename) # Make sure the file match with the name format (Job_(+d).txt)
                    if job_id_match: # If the format is match, get the jobId in the filename
                        job_id = job_id_match.group(1)
                    else: # Else set 'Unknown'
                        job_id = 'Unknown'

                    # Get the information in the filename
                    file_path = os.path.join(foldername, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                        # Extracting specific fields from the content
                        title = re.search(r'Title:\s*(.*)', content)
                        webindex = re.search(r'Webindex:\s*(.*)', content)
                        company = re.search(r'Company:\s*(.*)', content)
                        description = re.search(r'Description:\s*(.*)', content)
                        
                        # Put the information into the easiest format for next steps
                        job_info = {
                            'ID': job_id,
                            'Title': title.group(1).strip() if title else 'Unknown',
                            'Webindex': webindex.group(1).strip() if webindex else 'Unknown',
                            'Company': company.group(1).strip() if company else 'Unknown',
                            'Description': description.group(1).strip() if description else 'Unknown',
                            'Label': os.path.basename(foldername) # Get folder name as Label
                        }

                        job_infos.append(job_info)

    return job_infos

# Specify the root folder that contain all the job_ads folders
root_folder = r'.\.'
job_infors = get_job_infors(root_folder)

# %%
# Get all the description of the job_infors into an array
descriptions = []
for infor in job_infors:
    descriptions.append(infor['Description'])

# %%
# Get the stopwords list
with open(r'.\stopwords_en.txt', 'r') as file:
    stopwords = file.read().splitlines()

# %% [markdown]
# ### 1.2 Pre-processing data
# In this step, we started to perform all of the text preprocessing steps to complete all the requirements that were provided.

# %% [markdown]
# #### a. Tokenization
# In the beginning, we tokenize all the descriptions with the pattern that was provided.

# %%
# Tokenization with the giving pattern
tokens = []
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
tokenizer = RegexpTokenizer(pattern) 
for description in descriptions: # For every description in the collection array of description
    token = tokenizer.tokenize(description) # Tokenize that description
    tokens.append(token)

# %% [markdown]
# #### b. Lower-case

# %%
# Change all the words into lower format
tokens_lower = []
for tokene in tokens: # For every token array_list in the tokens
    token_list = [] # Create array token_list to store all tokens of 1 list
    for token in tokene: # For every token in the token array_list 
        token_list.append(token.lower()) # Lower the token and put it back to the token_list
    tokens_lower.append(token_list) # Get all the token_list into collection of token_lower

# %% [markdown]
# #### c. Remove words with length less than 2

# %%
tokens_length = []
for tokens in tokens_lower:
    token_list = []
    for token in tokens:
        if len(token) >= 2: # If length of the word high than 2, keep the word
            token_list.append(token)
    tokens_length.append(token_list)

# %% [markdown]
# #### d. Remove stopwords using the provided stop words list

# %%
tokens_without_stopwords = []
for tokens in tokens_length:
    token_list = []
    for token in tokens:
        if token not in stopwords: # If the word not in stopwords list, keep the word
            token_list.append(token)
    tokens_without_stopwords.append(token_list)

# %% [markdown]
# #### e. Remove the word that appears only once in the document collection, based on term frequency.

# %%
term_frequency = defaultdict(int) # Create a dictonary that contain the frequency of term
for tokens in tokens_without_stopwords:
    for token in tokens: 
        term_frequency[token] += 1 # +1 for the word if it appear

tokens_more_than_1 = []
token_len = []
for tokens in tokens_without_stopwords:
    tokens_filtered_freq = []
    for token in tokens:
        if term_frequency[token] > 1: # If the term_frequency higher than 1, keep the word
            tokens_filtered_freq.append(token)
    tokens_more_than_1.append(tokens_filtered_freq)


# %% [markdown]
# #### f. Remove the top 50 most frequent words based on document frequency.

# %%
document_frequency = defaultdict(int) # Create a dictonary that contain the document frequency

for document in tokens_more_than_1:
    unique_document_word = set(document) # Get all the word in a document unique
    for word in unique_document_word:
        document_frequency[word] += 1 # +1 for everytime the word appear in 1 document

more_than_50 = []
for word, count in document_frequency.items():
    more_than_50.append((word,count)) # Append the word and document_frequency count into more_than_50
    more_than_50.sort(key=lambda x: x[1], reverse=True) # Sort the more_than_50 from highest to lowest
    if len(more_than_50) > 50: # If this array length more than 50
        more_than_50.pop() # Remove the lowest count

# %%
# After get the list of highest document frequency, remove it in the tokens
final_list = []
for tokens in tokens_more_than_1:
    token_list = []
    for token in tokens:
        if token not in more_than_50: # If the word not in more_than_50, keep the word
            token_list.append(token)
    final_list.append(token_list)

# %% [markdown]
# #### g. Lemmatization

# %%
lemmatizer = WordNetLemmatizer()
lemmatized_list = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in final_list]

# %%
# After finishing the text pre-processing steps, get all the processed description back to job_infors
for i, description in enumerate(lemmatized_list):
    if i < len(job_infors):
        job_infors[i]['Description'] = description

# %% [markdown]
# #### Prepare Title vocabulary

# %%
# Get all the title of the job_infors into an array
titles = []
for infor in job_infors:
    titles.append(infor['Title'])

# %%
tokens = []
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
tokenizer = RegexpTokenizer(pattern) 
for title in titles: # For every title in the collection array of title
    token = tokenizer.tokenize(title) # Tokenize that title
    tokens.append(token)

# Turn every token into lowercase
tokens_lower_list = [[token.lower() for token in tokene] for tokene in tokens]
# Remove the stopwords
tokens_title_without_stopwords = [[token for token in tokens if token not in stopwords] for tokens in tokens_lower_list]

# %%
# After finishing the text pre-processing steps, get all the processed titles back to job_infors
for i, title in enumerate(tokens_title_without_stopwords):
    if i < len(job_infors):
        job_infors[i]['Title'] = title

# %% [markdown]
# ## Saving required outputs
# After finishing the text preprocessing for the descriptions in job advertisements, we save all the job ads into the preprocessed_job_ads.txt file and vocab.txt for vocabulary in the descriptions. We also create the vocab_title.txt for our following tasks.

# %%
# Function to export the information of job advertisement after pre-processing
def save_job_info(job_infors):
    header = "ID,  Title,  Webindex,  Company,  Description,  Label\n"

    lines = [header]

    for job_info in job_infors:
        # Join all the words into single string in Description
        sentence = ' '.join(job_info['Description'])
        job_info['Description'] = sentence
        # Join all the words into single string in Title
        sentence_title = ' '.join(job_info['Title'])
        job_info['Title'] = sentence_title
        # Create the file's content performance
        line = f"{job_info['ID']},  {job_info['Title']},  {job_info['Webindex']},  {job_info['Company']},  {job_info['Description']},  {job_info['Label']}\n"
        lines.append(line)
    return lines

# %%
content = save_job_info(job_infors)
# Write the new file based on the content
with open('preprocessed_job_ads.txt', 'w', encoding='utf-8') as output_file:
    output_file.writelines(content)

# %%
# Create vocab.txt file to store all Description vocabulary
vocabulary = []
for tokens in final_list:
    for token in tokens:
        if token not in vocabulary:
            vocabulary.append(token)
vocabulary.sort() # Sort to preform from A-Z
ids = 0

with open('vocab.txt', 'w') as file:
    for word in vocabulary:
        file.write(f'{word}:{ids}\n') # Included Id for each vocabulary
        ids+=1

# %%
# Create vocab_title.txt file to store all Title vocabulary
vocabulary_title = []
for tokens in tokens_title_without_stopwords:
    for token in tokens:
        if token not in vocabulary_title:
            vocabulary_title.append(token)
vocabulary_title.sort() # Sort to preform from A-Z
ids = 0

with open('vocab_title.txt', 'w') as file:
    for word in vocabulary_title:
        file.write(f'{word}:{ids}\n') # Included Id for each vocabulary
        ids+=1

# %% [markdown]
# ## Summary
# The job advertising' descriptions were successfully prepared for additional analysis and classification by completing all preprocessing tasks. Tokenization, lowercasing, and the removal of stop words, short words, uncommon keywords, and often used words have helped to focus the description section's attention on the relevant information.  The vocabulary of the description and the summary of job advertisement are provided for further tasks.


