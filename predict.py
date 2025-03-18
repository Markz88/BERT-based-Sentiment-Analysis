import argparse
import torch
from src.model import SentimentClassifier
from transformers import BertTokenizer
 
BERT_MODEL_NAME = "bert-base-uncased" # pre-trained BERT model
NUM_CLASSES = 3 # number of sentiment classes
CLASS_NAMES = ['negative', 'neutral', 'positive'] # sentiment classes {0: negative, 1: neutral, 2: positive}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device to use 

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME) # BERT tokenizer
model = SentimentClassifier(BERT_MODEL_NAME, NUM_CLASSES).to(DEVICE) # BERT sentiment classifier model

if DEVICE.type == "cpu":
    model.load_state_dict(torch.load('models/bert_sentiment_model.pth', map_location=DEVICE)) # load trained model
else:
    model.load_state_dict(torch.load('models/bert_sentiment_model.pth')) # load trained model
    
# Predict sentiment from text
def predict_sentiment(model, tokenizer, text, device, max_len=512):
    model = model.eval() # set model to evaluation mode
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move input tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict sentiment
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1) # get predicted sentiment
    
    return prediction.item() # return predicted sentiment

if __name__ == "__main__":
    # Parse command line arguments for input text
    parser = argparse.ArgumentParser(description="Predict sentiment from text")
    parser.add_argument("--text", type=str, required=True, help="Input text for sentiment analysis")
    args = parser.parse_args()
    
    # Predict sentiment
    sentiment = predict_sentiment(model, tokenizer, args.text, DEVICE)

    # Map predicted sentiment to class name
    if sentiment == 0:
        sentiment = CLASS_NAMES[0]
    elif sentiment == 1:
        sentiment = CLASS_NAMES[1]
    else:
        sentiment = CLASS_NAMES[2]
    
    print(f"Predicted Sentiment: {sentiment}")
