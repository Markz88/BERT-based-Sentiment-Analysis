import torch.nn as nn
from transformers import BertModel

# Create a sentiment classifier
class SentimentClassifier(nn.Module):
    # Constructor
    def __init__(self, bert_model_name, num_classes):
        # Call the constructor of the parent class
        super(SentimentClassifier, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Dropout layer
        self.drop = nn.Dropout(p=0.3)
        # Classification layer
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, dict_flag = False):
        # Get the last hidden state and pooled output
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids,  attention_mask=attention_mask, return_dict=dict_flag)
        output = self.drop(pooled_output) # Apply dropout on pooled output
        return self.out(output) # Get the output from the classifier