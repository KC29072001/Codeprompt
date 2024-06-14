
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from openai import OpenAI
import os
from dotenv import load_dotenv
import openai

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")



# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize OpenAI client
# client = OpenAI.api_key=API_KEY
openai.api_key = API_KEY


# Define preprocess_input function
def preprocess_input(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stopwords.words('english')]
    return tokens

# Define function to generate code
def generate_code(text):
    # Prompting the GPT-3 model with the natural language description
    prompt = f"Translate the following natural language description into code:\n\n{text}\n\nCode:"

    # Requesting code generation from GPT-3
    response = openai.chat.completions.create(
        messages=[{'role': 'user', 'content': text}],
        temperature=0,  
        model="gpt-3.5-turbo",
        max_tokens=500
    )


    return response.choices[0].message.content

    # Extracting the generated code from the response
    # return response.choices[0].message.content

# Streamlit UI
st.title("Code Generation from Text")

# Input box for the user to enter a sentence
user_input = st.text_input("Enter a natural language description:")

if user_input:
    # Preprocess input
    preprocessed_input = preprocess_input(user_input)
    preprocessed_input_str = ' '.join(preprocessed_input)
    
    # Generate code
    generated_code = generate_code(preprocessed_input_str)
    
    # Display generated code
    st.code(generated_code, language="python")
