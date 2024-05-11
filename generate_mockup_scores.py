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
columns = ["Input", "Set", "Question", "Star Rating"]
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
ratings = np.zeros((len(inputs), 1))
df = pd.DataFrame(
    np.hstack([inputs, sets, questions, ratings]), columns=columns)
# define the high and low scores
low_scores = np.random.randint(1, 4, int(len(df)/2))
high_scores = np.random.randint(3, 6, int(len(df)/2))
# populate score columns with low and high scores
idx_set1 = df[df['Set'] == 1].index
idx_set2 = df[df['Set'] == 2].index
df.loc[idx_set1, "Star Rating"] = high_scores
df.loc[idx_set2, "Star Rating"] = low_scores

for c in columns:
    df[c] = df[c].astype(np.int32)

df.to_csv("Mockup_Scores.csv", index=None)

"""
    Load the generated human feedback and add it is a column to the score dataframe
    Since the files are tab-delimiter - modify the pd.read_csv command to 
    accomodate
"""
feedback_files = glob("*Feedback.csv")
feedback_frame = pd.concat([pd.read_csv(f, delimiter='\t') for f in feedback_files]).reset_index(drop=True)
    
df_generated = pd.concat([df, feedback_frame['Feedback']], axis=1)

df_generated.to_csv("Generated_Mockup_Data.csv", index=None)
