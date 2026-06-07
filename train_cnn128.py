import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import multiprocessing
import io

class CNN128(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

if __name__ == "__main__":   
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True

    IMG_SIZE = 128
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # Τυχαία αναστροφή
        transforms.RandomRotation(10),           # Μικρή περιστροφή
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_train = datasets.ImageFolder("Cifake_Dataset/train")

    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(full_train, [train_size, val_size])

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = CNN128().to(device)
    criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
            xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        model.eval()
        val_running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
                xb, yb_labels = xb.to(device), yb.to(device)
                yb_loss = yb.float().unsqueeze(1).to(device) 
                outputs = model(xb)
                v_loss = criterion(outputs, yb_loss)
                val_running_loss += v_loss.item() * xb.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds.squeeze() == yb_labels).sum().item()
                total += yb_labels.size(0)

        epoch_loss = running_loss / train_size
        epoch_val_loss = val_running_loss / val_size
        val_acc = correct / total * 100

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "cnn128_best_model.pth")

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch}: Loss: {epoch_loss:.4f} | ValLoss: {epoch_val_loss:.4f} | ValAcc: {val_acc:.2f}%")

    print("\nΕκπαίδευση ολοκληρώθηκε!")

    plt.figure(figsize=(14, 6))
    epochs_range = range(1, EPOCHS + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, color='blue', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, color='orange', label='Validation Loss', linewidth=2)
    
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    plt.scatter(min_val_epoch, min_val_loss, color='red', zorder=5) 
    plt.annotate(f'Min: {min_val_loss:.4f}', 
                 xy=(min_val_epoch, min_val_loss), 
                 xytext=(min_val_epoch, min_val_loss + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title("Training & Validation Loss", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, color='green', label='Validation Accuracy', linewidth=2)
    plt.title("Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("cnn128_plot.png")
    print("Το διάγραμμα αποθηκεύτηκε ως: cnn128_plot.png")
    plt.show()