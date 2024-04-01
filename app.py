import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
from openai import OpenAI


import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# import os
# import streamlit as st
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from langchain.document_loaders import TextLoader
# from openai import OpenAI


# # Set NLTK data path relative to the root folder
# nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
# nltk.data.path.append(nltk_data_path)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': text}],
        temperature=0,  
        model="gpt-3.5-turbo",
        max_tokens=500
    )

    # Extracting the generated code from the response
    return response.choices[0].message.content

# Streamlit UI
st.title("Code Generation from Text")

# File uploader for the three text files
txt_file1 = st.file_uploader("Upload Text File 1", type=["txt"])
txt_file2 = st.file_uploader("Upload Text File 2", type=["txt"])
txt_file3 = st.file_uploader("Upload Text File 3", type=["txt"])

if txt_file1 and txt_file2 and txt_file3:
    # Load text from uploaded files
    text1 = txt_file1.read().decode("utf-8")
    text2 = txt_file2.read().decode("utf-8")
    text3 = txt_file3.read().decode("utf-8")

    # Preprocess and generate code for each text
    for text in [text1, text2, text3]:
        preprocessed_text = preprocess_input(text)
        preprocessed_text_str = ' '.join(preprocessed_text)
        generated_code = generate_code(preprocessed_text_str)
        
        # Display generated code
        st.code(generated_code, language="python")
