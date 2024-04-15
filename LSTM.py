import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = x.device
        h0, c0 = self.init_hidden(x.size(0), device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])  # Get last time step
        return out

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    

def train_model(model, train_data, criterion, optimizer, num_epochs):
    device = next(model.parameters()).device
    model.train()
    for epoch in range(num_epochs):
        for seq, labels in train_data:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} Loss: {loss.item()}')

