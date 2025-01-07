from transformers import AutoModelForSequenceClassification
from globals import model_name, num_labels

def get_model(device='cpu'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device=device)
    return model