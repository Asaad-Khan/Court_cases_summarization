# Optimized punishment info extraction (single shortest relevant sentence)
def extract_punishment_info(text):
    sentences = [sent.text for sent in sentenize(text)]
    keywords = ['назначил наказание', 'приговорил', 'наказание в виде', 'лет лишения свободы', 'условно', 'колонии']
    punishment_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    if punishment_sentences:
        # Return the shortest sentence to avoid lengthy appendage
        return min(punishment_sentences, key=len)
    return ''

# Optimized summarization function
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
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        max_length=80,  # Slightly shorter summary
        min_length=20,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Append only one short punishment sentence
    punishment_info = extract_punishment_info(cleaned_text)
    if punishment_info and punishment_info not in summary:
        summary += f" {punishment_info}"

    return summary
