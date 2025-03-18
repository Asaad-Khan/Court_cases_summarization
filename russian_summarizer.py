import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from razdel import sentenize
import nltk
import re
from bs4 import BeautifulSoup

nltk.download('stopwords')

# Load pre-trained Russian summarization model
model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Text cleaning and preprocessing function
def preprocess(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"ПРИГОВОР|Именем Российской Федерации", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^а-яА-ЯёЁ0-9\s]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

# Robust summarization function to avoid repetition
def summarize_russian(text):
    cleaned_text = preprocess(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        no_repeat_ngram_size=3,  # prevents phrase repetition
        length_penalty=2.0,
        max_length=100,
        min_length=30,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app configuration
st.set_page_config(page_title="Russian Court Case Summarizer", layout="wide")

# Streamlit UI
st.title("⚖️ Russian Court Case Summarizer")

user_text = st.text_area("Введите текст судебного дела:", height=300)

if st.button("Создать резюме"):
    if user_text.strip():
        with st.spinner('Создание резюме...'):
            summary = summarize_russian(user_text)
        st.subheader("📌 Резюме дела:")
        st.write(summary)
    else:
        st.error("Пожалуйста, введите текст судебного дела.")




