# %% [markdown]
# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Hong Thai Ngoc Ha, Dao Sy Trung Kien, Dao Quang Minh
# #### Student ID: S4060340, S3979613, S4015939
# 
# Date: 04/09/2024
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# In tasks 2 and 3, students need to create feature vectors for job advertisement descriptions and titles. Then, using the created vectors, build machine learning models to classify the category of a job advertisement text. Students should provide answers to two questions:
# 
# Q1: Language model comparisons
# 
# Q2: Does more information provide higher accuracy?

# %% [markdown]
# ## Importing libraries 

# %%
# Code to import libraries as you need in this assessment, e.g.,
import nltk
from nltk.tokenize import RegexpTokenizer
import os
import string
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

# %% [markdown]
# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions and Titles

# %% [markdown]
# ### 2.1 Load Vocabulary from Vocab Files and Extract Title, Description, and Webindex from Preprocessed Job Advertisement File

# %%
# Create a dict to store vocabulary of both titles and descriptions
vocab_all = {}
index = 1

# %% [markdown]
# a. Load the description vocabulary from the vocab.txt file

# %%
vocab_des = {}
with open('vocab.txt', 'r') as file:
    for line in file:
        word, idx = line.strip().split(':')
        vocab_des[word] = int(idx)
        vocab_all[word] = int(idx)
        index += 1

# %% [markdown]
# b. Load the title vocabulary from the vocab_title.txt file

# %%
vocab_title = {}
with open('vocab_title.txt', 'r') as file:
    for line in file:
        next_index = index
        word, idx = line.strip().split(':')
        vocab_title[word] = int(idx)
        if word not in vocab_all:
            vocab_all[word] = next_index
            next_index += 1

# %% [markdown]
# c. Extract descriptions, titles, and webindexes from Preprocessed Job Advertisement File

# %%
# Define lists to store the descriptions, titles, labels, and webindexes
descriptions = []
titles = []
webindexes = []
labels = []
all_doc = []

# Open and read the preprocessed_job_ads.txt file
with open('preprocessed_job_ads.txt', 'r') as file:
    # First, we skip the header
    next(file)
    # Next iterate through each line
    for line in file:
        # Split the line by commas
        fields = line.strip().split(', ')
        # Extract the 'title', 'webindex', 'labels', and 'description' fields
        if len(fields) >= 5:
            title = fields[1]
            webindex = fields[2]
            description = fields[4]
            label = fields[5]
            # Add the title, webindex, label, and description to the respective lists
            titles.append(title)
            webindexes.append(webindex)
            descriptions.append(description)
            labels.append(label)
            all_doc.append(title)
            all_doc.append(description)

# %% [markdown]
# ### 2.2 Generating Count Vectors

# %% [markdown]
# a. Descriptions Count Vector

# %%
# Create a CountVectorizer object with the vocabulary from the vocab dictionary
c_vec_des = CountVectorizer(vocabulary=vocab_des)

# Fit and transform descriptions using the c_vec_des
X_c_des = c_vec_des.fit_transform(descriptions)

# %% [markdown]
# b. Titles Count Vector

# %%
# Create a CountVectorizer object with the vocabulary from the vocab_title dictionary
c_vec_title = CountVectorizer(vocabulary=vocab_title)

# Fit and transform titles using the c_vec_title
X_c_title = c_vec_title.fit_transform(titles)

# %% [markdown]
# c. Titles and Descriptions Vector

# %%
# Clean vocabulary dict
vocab_all = set(vocab_all)

# Create a CountVectorizer object with the vocabulary from the vocab_all dictionary
c_vec_all = CountVectorizer(vocabulary=vocab_all)

# Fit and transform titles and description using the c_vec_all
X_c_all = c_vec_all.fit_transform(all_doc)

# %% [markdown]
# ### 2.3 Generating TF-IDF Vectors

# %% [markdown]
# a. Desccriptions TF-IDF Vector

# %%
# Create a TfidfVectorizer object with the vocabulary from the vocab dictionary
tfidf_vec_des = TfidfVectorizer(vocabulary=vocab_des)

# Fit and transform descriptions using the TfidfVectorizer
X_tfidf_des = tfidf_vec_des.fit_transform(descriptions)

# %% [markdown]
# b. Titles TF-IDF Vector

# %%
# Create a TfidfVectorizer object with the vocabulary from the vocab_title dictionary
tfidf_vec_title = TfidfVectorizer(vocabulary=vocab_title)

# Fit and transform titles using the TfidfVectorizer
X_tfidf_title = tfidf_vec_title.fit_transform(titles)

# %% [markdown]
# c. Combination TF-IDF Vector

# %%
# Create a TfidfVectorizer object with the vocabulary from the vocab_all dictionary
tfidf_vec_all = TfidfVectorizer(vocabulary=vocab_all)

# Fit and transform titles and descriptions using the TfidfVectorizer
X_tfidf_all = tfidf_vec_all.fit_transform(all_doc)

# %% [markdown]
# ### 2.4 Generating One-hot Vectors

# %% [markdown]
# a. Descriptions One-hot Vector

# %%
# Create a Binary CountVectorizer object with the vocabulary from the vocab dictionary
one_hot_vec_des = CountVectorizer(vocabulary=vocab_des, binary=True)

# Fit and transform the descriptions using the one_hot_vector
X_one_des = one_hot_vec_des.fit_transform(descriptions)

# %% [markdown]
# b. Titles One-hot Vector

# %%
# Create a Binary CountVectorizer object with the vocabulary from the vocab_title dictionary
one_hot_vec_title = CountVectorizer(vocabulary=vocab_title, binary=True)

# Fit and transform the titles using the CountVectorizer
X_one_tilte = one_hot_vec_title.fit_transform(titles)

# %% [markdown]
# c. Titles and Descriptions One-hot Vector

# %%
# Create a Binary CountVectorizer object with the vocabulary from the vocab_all dictionary
one_hot_vec_all = CountVectorizer(vocabulary=vocab_all, binary=True)

# Fit and transform the titles and descriptions using the CountVectorizer
X_one_all = one_hot_vec_all.fit_transform(all_doc)

# %% [markdown]
# ### 2.5 Generating Word2Vec Model

# %%
# Train the Word2Vec model on the tokenized descriptions
word2vecdes_model = Word2Vec(sentences=descriptions, vector_size=100, window=5, min_count=1, workers=4)

# %%
# Train the Word2Vec model on the tokenized titles
word2vectitle_model = Word2Vec(sentences=titles, vector_size=100, window=5, min_count=1, workers=4)

# %%
# Train the Word2Vec model on the tokenized titles and descriptions
word2vecall_model = Word2Vec(sentences=all_doc, vector_size=100, window=5, min_count=1, workers=4)

# %% [markdown]
# ### 2.6 Saving the Vector Representation

# %%
# Create a txt file to store the Count Vector representation of job advertisement descriptions
# With this following format: word_integer_index:word_freq
output_file = 'count_vectors.txt'
with open(output_file, 'w') as file:
    for i, webindex in enumerate(webindexes):
        sparse_row = X_c_des[i]
        non_zero = sparse_row.nonzero()[1]
        sparse_represent = []
        for word_integer_index in non_zero:
            word_freq = sparse_row[0, word_integer_index]
            sparse_represent.append(f"{word_integer_index}:{word_freq}")
        line = f"#{webindex}," + ','.join(sparse_represent) + '\n'
        file.write(line)

# %% [markdown]
# ## Task 3. Job Advertisement Classification

# %% [markdown]
# ## Split data into train and test sets

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
seed = 0

num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True)

# %%
def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# %%
def evaluate_based_on_kf(count_df, tfidf_df, onehot_df, num_of_folds):
    title_cv_df = pd.DataFrame(columns = ['count','tfidf','onehot'],index=range(num_of_folds))

    fold = 0
    for train_index, test_index in kf.split(list(range(0,len(labels)))):
        y_train = [labels[i] for i in train_index]
        y_test = [labels[i] for i in test_index]

        title_cv_df.loc[fold,'count'] = evaluate(count_df[train_index],count_df[test_index],y_train,y_test,seed)
        
        title_cv_df.loc[fold,'tfidf'] = evaluate(tfidf_df[train_index],tfidf_df[test_index],y_train,y_test,seed)

        title_cv_df.loc[fold,'onehot'] = evaluate(onehot_df[train_index],onehot_df[test_index],y_train,y_test,seed)
        
        fold +=1
    return title_cv_df

# %% [markdown]
# ## Classification using LogisticRegression on description

# %%
num_models = 3
cv_df = evaluate_based_on_kf(X_c_des, X_tfidf_des, X_one_des, num_folds)

# %%
cv_df

# %%
cv_df['tfidf'].mean()

# %% [markdown]
# ## Classification using LogisticRegression on Title

# %%
num_models = 3
title_cv_df = evaluate_based_on_kf(X_c_title, X_tfidf_title, X_one_tilte, num_folds)


# %%
title_cv_df

# %%
title_cv_df['tfidf'].mean()

# %% [markdown]
# ## Classification on title and description

# %%
num_models = 3
doc_cv_df = evaluate_based_on_kf(X_c_all, X_tfidf_all, X_one_all, num_folds)


# %%
doc_cv_df

# %%
doc_cv_df['tfidf'].mean()

# %% [markdown]
# ## Summary
# ### Q1: Language model comparisons:
# From 'cv_df' DataFrame, we can clearly see that the tfidf vector gives the best result (highest accuracy is 0.85 while average is -.79).
# 
# The model accuracy on 5-fold test is the highest on every fold. The onehot vector and count vector is pretty similar but the onehot vector is slightly better.
# ### Q2: Impact of amount of information on the accuracy:
# Different approach of the data gives us different vocabulary to work with. In this assignment, we created 3 vocabulary based on 3 approaches:
# 1. Build vocabulary based on titles
# 2. Build vocabulary based on descriptions
# 3. Build vocabulary based on titles AND descriptions
# 
# The results of approaches (1) and (2) are not very differnt from each other (best model works on tfidf vector, the average accuracy is nearly 0.8) with the apprach (2) is slightly better.
# 
# The approach (3) has the most information (gather text from both titles and descriptions) but it's performance is not as good as the other two.
# 
# We can see that the descriptions can generate bigger vocabulary that the titles and their combination will generate an even bigger one. But the accuracy only improves for the case of titles to descriptions
# 
# From the result above, it is safe to say that more information is not always improve the accuracy
# 
# 

# %% [markdown]
# 


