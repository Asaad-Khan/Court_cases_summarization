import streamlit as st
from langchain.llms.base import LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bs4 import BeautifulSoup
import re

# Define a LangChain Wrapper for RuT5
class RuT5Summarizer(LLM):
    def __init__(self, model_name="ai-forever/ruT5-base"):
        """
        Loads the RuT5 model for Russian legal text summarization.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def _call(self, prompt: str, stop=None):
        """
        Processes input text and returns the summary.
        """
        # Preprocess the input for T5
        inputs = self.tokenizer(
            f"суммаризировать: {prompt}",
            return_tensors="pt",
            truncation=True,
            max_length=1024  # RuT5 supports 1024 tokens
        )

        # Generate summary with length constraints
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=300,  # Adjust based on needed summary length
            min_length=100,
            num_beams=5,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the LangChain-based RuT5 Summarizer
summarizer = RuT5Summarizer()

# Text Preprocessing
def preprocess_text(text):
    """
    Prepares legal text for summarization by:
    - Removing extra spaces/newlines
    - Adding the required 'суммаризировать:' prefix for T5 models
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"ПРИГОВОР|Именем Российской Федерации", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^а-яА-ЯёЁ0-9.,;()\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Russian Court Case Summarizer (RuT5 + LangChain)", layout="wide")

st.title("⚖️ Russian Court Case Summarizer (RuT5 + LangChain)")

user_text = st.text_area("Enter the court case text:", height=300)

if st.button("Generate Summary"):
    if user_text.strip():
        with st.spinner('Generating summary...'):
            processed_text = preprocess_text(user_text)
            summary = summarizer(processed_text)  # ✅ LangChain-based summarization

        st.subheader("📌 Case Summary:")
        st.write(summary)
    else:
        st.error("Please enter court case text.")
