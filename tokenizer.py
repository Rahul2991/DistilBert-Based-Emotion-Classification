from transformers import AutoTokenizer
from globals import model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)

get_tokenizer = lambda: tokenizer

def tokenize_function(inp):
    return tokenizer(inp['text'], truncation=True, padding='max_length', max_length=512)

