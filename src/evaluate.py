import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model
def evaluate_model(model, data_loader, device, phase = 'val'):
    # Set the model to evaluation mode
    model = model.eval()
    predictions = [] # Predictions
    actual_labels = [] # Actual labels
    
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the data
        for batch in tqdm(data_loader, leave=False, total=len(data_loader)):
            # Get the input data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Get the model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1) # Get the predictions
            
            predictions.extend(preds.cpu().numpy()) # Append the predictions
            actual_labels.extend(labels.cpu().numpy()) # Append the actual labels
    
    if phase == 'test': # If the phase is test, return the accuracy and classification report over the test set
        acc = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions)
        return acc, report
    else: # If the phase is val, return the accuracy over the validation set
        acc = accuracy_score(actual_labels, predictions)
        return acc
    
    
