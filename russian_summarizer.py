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

# Extract key case details
def extract_case_details(text):
    sentences = [sent.text for sent in sentenize(text)]
    
    # Search for the defendant's name
    defendant_match = re.search(r'в отношении\s+([А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+)', text)
    defendant = defendant_match.group(1) if defendant_match else None

    # Search for the charges (Ugolovny Kodeks reference)
    charges_match = re.findall(r'ст\.\s*\d+\s*ч\.\s*\d+', text)
    charges = ', '.join(charges_match) if charges_match else None

    return defendant, charges

# Extract punishment details (years of sentence)
def extract_punishment_info(text):
    sentences = [sent.text for sent in sentenize(text)]
    keywords = ['назначил наказание', 'приговорил', 'наказание в виде', 'лет лишения свободы', 'условно', 'колонии']
    punishment_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    
    if punishment_sentences:
        return min(punishment_sentences, key=len)
    return ''

# Optimized summarization function
def summarize_russian(text):
    cleaned_text = preprocess(text)
    
    # Extract key case details
    defendant, charges = extract_case_details(cleaned_text)
    
    # Limit input to 512 tokens (prevents extreme-length summaries)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Adjust summary length dynamically
    input_word_count = len(cleaned_text.split())
    
    if input_word_count < 100:
        max_summary_length = 50  # Shorter for short inputs
    elif input_word_count < 300:
        max_summary_length = 70  # Medium-length texts
    else:
        max_summary_length = 90  # Long texts must stay concise
    
    # Generate summary with enforced max length
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.5,
        max_length=max_summary_length,
        min_length=20,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Extract punishment information
    punishment_info = extract_punishment_info(cleaned_text)

    # Build final summary dynamically
    final_summary = summary
    if defendant and charges:
        final_summary = f"{defendant} is charged under {charges}. {summary}"
    elif defendant:
        final_summary = f"{defendant} is involved in the case. {summary}"
    elif charges:
        final_summary = f"The case involves charges under {charges}. {summary}"
    
    if punishment_info:
        final_summary += f" {punishment_info}"
    
    return final_summary

# Streamlit app configuration
st.set_page_config(page_title="Russian Court Case Summarizer", layout="wide")

# Streamlit UI
st.title("⚖️ Russian Court Case Summarizer")

user_text = st.text_area("Enter the court case text:", height=300)

if st.button("Generate Summary"):
    if user_text.strip():
        input_word_count = len(user_text.split())
        with st.spinner('Generating summary...'):
            summary = summarize_russian(user_text)
        output_word_count = len(summary.split())
        
        st.subheader("📌 Case Summary:")
        st.write(summary)
        
        st.write(f"**Word count in original text:** {input_word_count}")
        st.write(f"**Word count in summary:** {output_word_count}")
    else:
        st.error("Please enter court case text.")
