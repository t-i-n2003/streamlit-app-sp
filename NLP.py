from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from underthesea import word_tokenize

# Dùng mô hình đã fine-tuned cho sentiment analysis
model = "justdoit2111/phobert-base-vietnamese-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model)
phobert = AutoModelForSequenceClassification.from_pretrained(model)
phobert.eval()

def preprocess(text):
    return word_tokenize(text, format="text")

def predict_label(text):
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = phobert(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    label = torch.argmax(probs, dim=1).item()

    return label, probs[0].tolist()
