from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, argparse, os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def preprocess_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    return inputs

def predict_emotion(text):
    inputs = preprocess_text(text).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class = torch.argmax(logits, dim=1).item()
        
    predicted_label = label_names[predicted_class]
    return predicted_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for DistilBERT Fine-tuning on Emotion Classification Dataset dair-ai/emotion")
    parser.add_argument("-m", "--model_path", type=str, default="./results", help="Model Directory (default: ./results)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.model_path): 
        print('Model path is invalid.')
        exit(1)
    
    text = None
    
    print('Loading Model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    print('Loading Model Successful')

    print('Enter text to predict emotion (type EXIT to quit):')
    while text != 'EXIT':
        text = input("Enter text to predict: ")
        if text == 'EXIT': break
        predicted_emotion = predict_emotion(text)
        print(f"Predicted emotion: {predicted_emotion}")