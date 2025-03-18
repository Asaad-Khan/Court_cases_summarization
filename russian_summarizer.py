import streamlit as st
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
import torch
import re
from bs4 import BeautifulSoup

# Load FRED-T5 Model & Tokenizer
model_name = "Sberbank-AI/FRED-T5-1.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Text Cleaning & Preprocessing
def preprocess_text(text):
    """
    Prepares legal text for summarization by:
    - Removing extra spaces/newlines
    - Adding the T5 required 'summarize:' prefix
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"ПРИГОВОР|Именем Российской Федерации", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^а-яА-ЯёЁ0-9\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return f"summarize: {text}"

# Summarization Function
def summarize_fred_t5(text):
    """
    Summarizes Russian court case texts using FRED-T5-1.7B.
    """
    processed_text = preprocess_text(text)
    
    # Tokenize input
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Max token limit for FRED-T5
    )
    
    # Generate summary with strict length control
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,  # Control summary length
        min_length=100,
        num_beams=5,  # Beam search for better quality
        length_penalty=1.1,  # Lower forces shorter summaries
        no_repeat_ngram_size=3,  # Prevents redundancy
        early_stopping=True
    )
    
    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App UI
st.set_page_config(page_title="Russian Court Case Summarizer (FRED-T5)", layout="wide")

st.title("⚖️ Russian Court Case Summarizer (FRED-T5)")

user_text = st.text_area("Enter the court case text:", height=300)

if st.button("Generate Summary"):
    if user_text.strip():
        with st.spinner('Generating summary...'):
            summary = summarize_fred_t5(user_text)
        
        st.subheader("📌 Case Summary:")
        st.write(summary)
    else:
        st.error("Please enter court case text.")
