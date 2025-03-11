import torch
from tqdm import tqdm

# Function to train the model for one epoch
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train() # set model to training mode
    total_loss = 0 # total loss for the epoch
    correct_predictions = 0 # number of correct predictions
    
    # iterate over the data loader
    for batch in tqdm(data_loader, leave=False, total=len(data_loader)):
        input_ids = batch['input_ids'].to(device) # move input to device
        attention_mask = batch['attention_mask'].to(device) # move attention mask to device
        labels = batch['label'].to(device) # move labels to device
        
        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels) # calculate loss
        total_loss += loss.item() # accumulate loss
        
        # calculate the number of correct predictions
        _, preds = torch.max(outputs, dim=1) # get the predictions
        correct_predictions += torch.sum(preds == labels) # accumulate correct predictions
    
        # backpropagation
        optimizer.zero_grad() # clear the gradients
        loss.backward() # calculate gradients
        optimizer.step() # update the weights
        scheduler.step() # update the learning rate
    
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)
