import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from razdel import sentenize
import torch
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk

nltk.download('stopwords')

# Load summarization model (RuT5 Gazeta)
model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Text cleaning function
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"(К\s*делу\s*№\s*)?(П\s*Р\s*И\s*Г\s*О\s*В\s*О\s*Р)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^а-яА-ЯёЁ\s]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

# Summarization function
def summarize_russian(text):
    cleaned_text = preprocess_text(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=150,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit interface
st.set_page_config(page_title="Russian Court Case Summarizer", layout="wide")

st.title("⚖️ Russian Court Case Summarizer")

user_text = st.text_area("Введите текст судебного дела:", height=250)

if st.button("Создать резюме"):
    if user_text.strip():
        with st.spinner('Создание резюме...'):
            summary = summarize_russian(user_text)
        st.subheader("📝 Резюме дела:")
        st.write(summary)
    else:
        st.error("Пожалуйста, введите текст для резюмирования.")
