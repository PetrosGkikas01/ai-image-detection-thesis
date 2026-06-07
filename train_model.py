import torch
import torch.nn as nn
import torch.optim as optim 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
import joblib

data = pd.read_csv("features.csv") 

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 

scaler = StandardScaler() 
X = scaler.fit_transform(X) 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

class my_first_FFNN(nn.Module):
    def __init__(self, input_dim):
        super(my_first_FFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),         
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = my_first_FFNN(input_dim)

criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001) 

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf') 
epochs = 2000

for epoch in range(epochs):
    model.train() 
    optimizer.zero_grad() 

    outputs = model(X_train) 
    loss = criterion(outputs, y_train) 
    loss.backward() 
    optimizer.step() 

    model.eval() 
    with torch.no_grad(): 
        val_outputs = model(X_val)
        v_loss = criterion(val_outputs, y_val) 
        
        val_preds = (val_outputs > 0.5).float() 
        val_acc = (val_preds.eq(y_val).sum() / y_val.shape[0]).item() * 100 

    if v_loss.item() < best_val_loss:
        best_val_loss = v_loss.item()
        torch.save(model.state_dict(), "my_first_ffnn_best_model.pth")
        

    train_losses.append(loss.item())
    val_losses.append(v_loss.item())
    val_accuracies.append(val_acc)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {v_loss.item():.4f}, Val Acc: {val_acc:.2f}%")


joblib.dump(scaler, "scaler.pkl")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, color='blue', label='Training Loss')
plt.plot(val_losses, color='orange', label='Validation Loss')

min_v_loss = min(val_losses)
min_v_epoch = val_losses.index(min_v_loss)
plt.scatter(min_v_epoch, min_v_loss, color='red', zorder=5)
plt.annotate(f'Min Val Loss: {min_v_loss:.4f}', 
             xy=(min_v_epoch, min_v_loss), 
             xytext=(min_v_epoch, min_v_loss + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.title("FFNN Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, color='green', label='Validation Accuracy')
plt.title("FFNN Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("ffnn_training_plot.png")
plt.show()

print(f"Εκπαίδευση ολοκληρώθηκε. Το καλύτερο μοντέλο αποθηκεύτηκε με Val Loss: {best_val_loss:.4f}")