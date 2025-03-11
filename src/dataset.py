import torch
from torch.utils.data import Dataset, DataLoader

# Create a custom dataset
class SentimentDataset(Dataset):
    # Constructor
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Length of the dataset
    def __len__(self):
        return len(self.texts)

    # Get an item from the dataset
    def __getitem__(self, item):
        # Get the text and label
        text = str(self.texts[item])
        label = self.labels[item]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        # Return the dictionary containing the input_ids, attention_mask, and label
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create a data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    # Create a dataset
    dataset = SentimentDataset(
        texts=df.content.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    # Return a data loader
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)