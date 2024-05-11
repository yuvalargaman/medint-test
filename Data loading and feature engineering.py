"""
    Data loading and feature engineering
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from glob import glob
stop_words = set(stopwords.words('english'))

def preprocess(document):
    # Tokenize text
    words = word_tokenize(document)
    # Remove stopwords and non-alphanumeric characters
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

def preprocess_mockup(fname):
    with open(fname, 'r') as file:
        c = [line.strip() for line in file.readlines()]
    input_string = c[c.index("Input:")+1].strip()
    initial_question = c[c.index("Initial question:")+1]
    idx_set1 = c.index("Set 1:")
    idx_set2 = c.index("Set 2:")
    questions = c[idx_set1+1:idx_set2] + c[idx_set2+1:]
    separated_input = [input_string, initial_question]
    processed_input = ' '.join([preprocess(d) for d in separated_input])
    processed_questions = [preprocess(q)[2:] for q in questions]
    return processed_input, processed_questions

"""
    The following is a pseudocode of my approach towards creating the feature vectors.
    For each mockup file:
        Preprocess the input and the questions
        For each question:
            Calculate TF-IDF or embeddings vectors of the input vs the question
            Calculate the cosine similarity between the vectors
"""

all_tfidf_similarities = []
all_embedding_similarities = []

for f in glob("Mockup_*.txt"):
    inputs, questions = preprocess_mockup(f)
    
    # Feature extraction
    # TF-IDF Vectors
    similarities = []
    tfidf_results = []
    features = []
    for q in questions:
        vectorizer = TfidfVectorizer()
        tfidf_result = vectorizer.fit_transform([inputs]+[q])    
        tfidf_matrix = tfidf_result.toarray()
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarities.append(similarity)
        tfidf_results.append(tfidf_matrix)
    all_tfidf_similarities.append(np.array(similarities).reshape(-1, 1))
    
    """ 
        Using embeddings instead
    """
    embeddings = []
    embedding_similarities = []
    for q in questions:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embedding = model.encode([inputs, q])
        embeddings.append(doc_embedding)
        cosine_sim = cosine_similarity([doc_embedding[0], doc_embedding[1]])
        embedding_similarities.append(cosine_sim[0][1])
    all_embedding_similarities.append(np.array(embedding_similarities).reshape(-1, 1))
    
all_tfidf_similarities = np.vstack(all_tfidf_similarities)
all_embedding_similarities = np.vstack(all_embedding_similarities)

df_similarities = pd.DataFrame(np.hstack([all_tfidf_similarities, all_embedding_similarities]), columns=['TF-IDF Similarities', 'Embedding Similarities'])

df_similarities.to_csv('Mockup_Question_to_Input_Similarities.csv', index=None)
    
"""
    I suggest using semtiment analysis to add star ratings and free human feedback as features to the evaluation model.
    Lower star ratings correpond to lower valence of a question to the input and initial question, so they can take up sentiment values such as:
        1 - very negative
        2 - negative
        3 - neutral
        4 - positive
        5 - very positive
    Similarly, a sentiment analysis model can be run on the human feedback in order to derive the above feedback valences.
    Then, it could be possible to use each of the valence vectors as features, or calculate the difference between the star and feedback valences to derive an agreement vector which would be used as a feature.
"""

    