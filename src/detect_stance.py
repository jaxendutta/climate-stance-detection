import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_data = pd.read_csv('data/processed/train.csv')
val_data = pd.read_csv('data/processed/val.csv')
test_data = pd.read_csv('data/processed/test.csv')

# Define dataset class
class StanceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['processed_text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(self.data.iloc[index]['stance'], dtype=torch.long)
        }

# Initialize tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)
model.to(device)

# Prepare datasets
train_dataset = StanceDataset(train_data, tokenizer, max_len=128)
val_dataset = StanceDataset(val_data, tokenizer, max_len=128)
test_dataset = StanceDataset(test_data, tokenizer, max_len=128)

# Prepare dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    return accuracy_score(true_labels, predictions), f1_score(true_labels, predictions, average='weighted')

# Training
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")
    
    val_accuracy, val_f1 = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

# Final evaluation on test set
test_accuracy, test_f1 = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")

# Save the model
torch.save(model.state_dict(), 'stance_detection_model.pth')
print("Model saved successfully.")
