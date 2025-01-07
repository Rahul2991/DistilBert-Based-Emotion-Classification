from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
from tokenizer import tokenize_function
import numpy as np
import torch
    
def get_dataset(device='cpu', only_class_weight=False):
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    
    # Compute class weights
    labels = dataset['train']['label']
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(device)
    
    if only_class_weight: return class_weights

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']
    
    return train_dataset, val_dataset, test_dataset