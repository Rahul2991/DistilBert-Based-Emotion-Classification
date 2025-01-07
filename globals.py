import torch

model_name = 'distilbert-base-uncased'
num_labels = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'