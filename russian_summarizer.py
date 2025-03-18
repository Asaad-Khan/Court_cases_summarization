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

# Optimized punishment info extraction (single shortest relevant sentence)
def extract_punishment_info(text):
    sentences = [sent.text for sent in sentenize(text)]
    keywords = ['назначил наказание', 'приговорил', 'наказание в виде', 'лет лишения свободы', 'условно', 'колонии']
    punishment_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    if punishment_sentences:
        # Return the shortest sentence to avoid lengthy appendage
        return min(punishment_sentences, key=len)
    return ''

def summarize_russian(text):
    cleaned_text = preprocess_text(text)
    
    # Limit input to 512 tokens (prevents extreme-length summaries)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Get word count of the cleaned input
    input_word_count = len(cleaned_text.split())

    # Dynamically set summary length limits
    if input_word_count < 100:
        max_summary_length = 50  # Short for short inputs
    elif input_word_count < 300:
        max_summary_length = 70  # Medium-length texts
    else:
        max_summary_length = 100  # Hard cap on long summaries

    # Generate summary with enforced max length
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.2,  # Lower value forces shorter summaries
        max_length=max_summary_length,
        min_length=20,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Extract punishment information
    punishment_info = extract_punishment_info(cleaned_text)

    # Construct final summary (only in Russian)
    final_summary = summary
    if punishment_info:
        final_summary += f" {punishment_info}"
    
    return final_summary.strip()

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
