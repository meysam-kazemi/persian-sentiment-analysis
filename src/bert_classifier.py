import torch
from torch import nn
from tqdm import tqdm

class BertClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        """
        A simple classifier on top of BERT embeddings.
        Args:
            hidden_size (int): The size of BERT's output embeddings (768 for base models).
            num_classes (int): The number of sentiment classes.
        """
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def train_bert_classifier(model, dataloader, device):
    classifier = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
    epochs = 2
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for batch_embeddings, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
    return model

