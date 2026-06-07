import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os
import multiprocessing


class HybridDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.features = self.df.iloc[:, :-2].values
        self.filenames = self.df["filename"].values
        self.labels = self.df["label"].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        folder = "REAL" if label == 0 else "FAKE"
        img_path = os.path.join(self.image_root, folder, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return image, features, label

class HybridCNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.cnn = nn.Sequential(
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

        self.cnn_fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU()
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, features):
        x = self.cnn(image)
        x = x.flatten(1)
        x = self.cnn_fc(x)
        f = self.feature_fc(features)
        combined = torch.cat((x, f), dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.cuda.empty_cache()

    IMG_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    FEATURE_DIM = 18
    VAL_SPLIT = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_dataset = HybridDataset(
        csv_file="features_train_3rd.csv",
        image_root="Cifake_Dataset/train",
        transform=None 
    )

    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_subset.dataset.transform = train_transform 
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridCNN(FEATURE_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, features = images.to(device), features.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, features, labels in val_loader:
                images, features = images.to(device), features.to(device)
                labels_loss = labels.unsqueeze(1).to(device)
                labels_acc = labels.to(device)

                outputs = model(images, features)
                v_loss = criterion(outputs, labels_loss)
                val_running_loss += v_loss.item() * images.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds.squeeze() == labels_acc).sum().item()
                total += labels_acc.size(0)

        epoch_val_loss = val_running_loss / val_size
        val_acc = 100 * correct / total
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "hybrid_best_model.pth")
            print(f"--> Best model saved at epoch {epoch}")

        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        print(f"Loss: {epoch_loss:.4f} | ValLoss: {epoch_val_loss:.4f} | ValAcc: {val_acc:.2f}%")

    print("\nHybrid training complete!")

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

    plt.title("Hybrid Model: Training & Validation Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, color='green', label='Validation Accuracy', linewidth=2)
    plt.title("Hybrid Model: Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("hybrid_plot.png")
    print("Το διάγραμμα αποθηκεύτηκε ως: hybrid_plot.png")
    plt.show()