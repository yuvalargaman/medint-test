"""
    Scripts for Part 2: Scripting and Implementation
"""
# %% Imports
import os
import re
from glob import glob
import numpy as np
import pandas as pd

# %% Preprocessing
"""
    Ten cases were generated manually using Ollama running Llama3, or
    by using ChatGPT 4.0 . First, the model was provided with several real-life
    cases which I had encountered, and an initial question.
    During this stage, following each input+question, the model was asked to 
    derive 2 sets of questions from the initial question:
        Set 1 - questions whos answers could potentially add information to the
        initial question and make the answer more concise. Star ratings in this
        case would range from 3 to 5.
        Set 2 - questions within or without the context of the initial question
        and input case, in varying degrees of vagueness. The star ratings
        in this case range from 1 to 3
    The star ratings and human-provided feedback would be randomly generated.
"""

"""
    Create the dataframe that will be used to map the scores to each
    input-set-question 
"""
columns = ["Input", "Set", "Question", "Star Rating",
           "Human-Context", "Human-Specificity", "Human-Relevance"]
n_questions = 8
n_sets = 2
overall_questions = n_questions*n_sets
n_inputs = 10
inputs = np.vstack([np.repeat(ii+1, overall_questions).reshape(-1, 1)
                   for ii in np.arange(n_inputs)])
sets = np.vstack([np.concatenate(
    [[ii+1]*n_questions for ii in np.arange(n_sets)]).reshape(-1, 1)]*n_inputs)
questions = np.vstack(
    [np.arange(1, n_questions+1).reshape(-1, 1)]*n_inputs*n_sets)
ratings = np.zeros((len(inputs), 4))
df = pd.DataFrame(
    np.hstack([inputs, sets, questions, ratings]), columns=columns)
# define the high and low scores
low_scores = np.random.randint(1, 4, size=(int(len(df)/2), 4))
high_scores = np.random.randint(3, 5, size=(int(len(df)/2), 4))
# populate score columns with low and high scores
idx_set1 = df[df['Set'] == 1].index
idx_set2 = df[df['Set'] == 2].index
df.iloc[idx_set1, 3:] = high_scores
df.iloc[idx_set2, 3:] = low_scores

