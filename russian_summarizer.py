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
        no_repeat_ngram_size=3,  # <-- prevents repeating phrases
        length_penalty=2.0,
        max_length=100,
        min_length=30,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



