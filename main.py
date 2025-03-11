import argparse
import torch
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.dataset import create_data_loader
from src.model import SentimentClassifier
from src.train import train_epoch
from src.evaluate import evaluate_model

# Configuration
BERT_MODEL_NAME = 'bert-base-uncased' # pre-trained BERT model
MAX_LEN = 160 # max length of input text
BATCH_SIZE = 16 # batch size
EPOCHS = 10 # number of training epochs
CLASS_NAMES = ['negative', 'neutral', 'positive'] # sentiment classes {0: negative, 1: neutral, 2: positive}
NUM_CLASSES = len(CLASS_NAMES) # number of sentiment classes
RANDOM_SEED = 42 # random seed for reproducibility
np.random.seed(RANDOM_SEED) # set random seed for numpy
torch.manual_seed(RANDOM_SEED) # set random seed for torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # device to use for training

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME) # BERT tokenizer
model = SentimentClassifier(BERT_MODEL_NAME, NUM_CLASSES).to(DEVICE) # BERT sentiment classifier model

# Load train, validation, and test data
df_train = pd.read_csv('data/train.csv')
train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

df_val = pd.read_csv('data/val.csv')
val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

df_test = pd.read_csv('data/test.csv')
test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# Optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS # total training iterations

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

# Best accuracy
best_accuracy = 0

if __name__ == '__main__':
    # Parse command line arguments for input mode
    parser = argparse.ArgumentParser(description="Predict sentiment from text")
    parser.add_argument("--mode", type=str, required=True, help="Input mode: \"train\" or \"evaluate\"")
    args = parser.parse_args()
    
    # Training mode
    if args.mode == 'train':
        # Training loop
        for epoch in range(EPOCHS):
            # Training
            train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, scheduler)
            print(f'Epoch {epoch+1}: Train Loss {train_loss}, Train Accuracy {train_acc}')

            # Validation
            val_acc = evaluate_model(model, val_loader, DEVICE, phase='val')
            print(f'Epoch {epoch+1}: Validation accuracy: {val_acc}\n')

            # Save best model based on validation accuracy
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                if not os.path.exists('models'):
                    os.makedirs('models')
                
                torch.save(model.state_dict(), 'models/bert_sentiment_model.pth') # save model
    else:
        # Evaluate mode
        model.load_state_dict(torch.load('models/bert_sentiment_model.pth'))
        eval_acc, eval_report = evaluate_model(model, test_loader, DEVICE, phase='test')
        print(f'Accuracy: {eval_acc:.2f}\n\n{eval_report}')
