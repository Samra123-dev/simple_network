import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create dataset and dataloader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define the neural network with 5 layers
class DeepNet(nn.Module):
    def __init__(self, input_size=2):
        super(DeepNet, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        # Layer 2
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.25)
        
        # Layer 3
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.2)
        
        # Layer 4
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.dropout4 = nn.Dropout(0.15)
        
        # Output layer
        self.fc5 = nn.Linear(8, 2)
        
    def forward(self, x):
        # Layer 1 with ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2 with tanh
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        
        # Layer 3 with ReLU
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Layer 4 with tanh
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.tanh(x)
        x = self.dropout4(x)
        
        # Output layer with log softmax
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# Initialize the model, loss function and optimizer
model = DeepNet()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
def train_model(model, train_loader, test_loader, epochs=150):
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_losses

# Train the model
train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=150)

# Plot the training and test losses
plt.figure(figsize=(5, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.title('Training and Test Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
 """
"""
# Plot the training and test losses with custom colors
plt.figure(figsize=(5, 5))
plt.plot(train_losses, 
         label='Training loss', 
         color='#1f77b4',  # Muted blue
         linewidth=2,
         alpha=0.8)
plt.plot(test_losses, 
         label='Test loss', 
         color='#ff7f0e',  # Safety orange
         linewidth=2,
         alpha=0.8,
         linestyle='--')  # Dashed line for test loss

# Customize the plot appearance
plt.legend(framealpha=1, shadow=True)
plt.title('Training and Test Loss Over Epochs', pad=20)
plt.xlabel('Epochs', labelpad=10)
plt.ylabel('Loss', labelpad=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Customize the background and spines
ax = plt.gca()
ax.set_facecolor('#f8f8f8')  # Light gray background
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
# Print model architecture
print("\nModel Architecture:")
print(model)