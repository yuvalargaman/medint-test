"""
    Generate free-human feedback for each question in each mockup input
"""

import os
import requests
from glob import glob

mockups = glob("Mockup*.txt")

def load_text(fname):
    with open(fname, 'r') as f:
        return f.read()

def query_ollama(prompt, url="http://localhost:11434/api/generate", model="llama3", stream=False):
    payload = {"model":model, "prompt":prompt, "stream":stream}
    response = requests.post(url, json=payload)
    return response

pre_prompt = "Based in the input and initial question in the prompt, generate a critical human feedback responses for each of the questions in each of the sets: "
mockup = load_text('Mockup_01.txt')
prompt = '\n'.join([pre_prompt, mockup])

result = query_ollama(prompt)
