import torch
import torch.nn as nn
import pandas as pd
import joblib # Φορτωση scaler που αποθηκευτηκε στην εκπαιδευση
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #για υπολογισμό Accuracy, Precision, Recall, F1 Score

data_test = pd.read_csv("features_test.csv")

X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values

scaler = joblib.load("scaler.pkl")
X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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


input_dim = X_test.shape[1]
model = my_first_FFNN(input_dim)
model.load_state_dict(torch.load("my_first_ffnn_best_model.pth", map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs > 0.5).float()


y_true_np = y_test.numpy()
y_pred_np = y_pred.numpy()

acc = accuracy_score(y_true_np, y_pred_np) * 100 # Σωστες προβλεψεις
prec = precision_score(y_true_np, y_pred_np) # ποσες AI προβλεψεις ηταν οντως AI
rec = recall_score(y_true_np, y_pred_np) # ποσες AI generated εικονες βρηκαμε
f1 = f1_score(y_true_np, y_pred_np) # Μ.Ο. των Precision και Recall

# Εκτυπωση αποτελεσματων
print(f"\nΑΠΟΤΕΛΕΣΜΑΤΑ TESTING:")
print(f"Accuracy:  {acc:.2f}%")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1 Score:  {f1:.2f}")


