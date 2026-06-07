import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import multiprocessing
from PIL import Image, ImageChops, ImageEnhance
import io

class ELATransform:
    def __init__(self, quality=90, scale=15.0):
        self.quality = quality
        self.scale = scale

    def __call__(self, img):
        original = img.convert('RGB')
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        ela_img = ImageChops.difference(original, resaved)
        ela_img = ImageEnhance.Brightness(ela_img).enhance(self.scale)
        return ela_img

class MultiStreamDataset(Dataset):
    def __init__(self, root_dir, transform_rgb=None, transform_ela=None):
        self.base_dataset = datasets.ImageFolder(root_dir)
        self.transform_rgb = transform_rgb
        self.transform_ela = transform_ela
        self.ela_proc = ELATransform(quality=90, scale=15.0)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        

        ela_img = self.ela_proc(img)
        
        img_rgb = self.transform_rgb(img)
        img_ela = self.transform_ela(ela_img)
            
        return img_rgb, img_ela, label

class MultiStreamCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.ela_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, ela):
        feat_rgb = self.rgb_stream(rgb).flatten(1)
        feat_ela = self.ela_stream(ela).flatten(1)
        combined = torch.cat((feat_rgb, feat_ela), dim=1)
        return self.fc(combined)

if __name__ == "__main__":   
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True

    IMG_SIZE, BATCH_SIZE, LR, EPOCHS = 128, 64, 0.001, 50
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

    full_dataset = MultiStreamDataset("Cifake_Dataset/train", train_transform, train_transform)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    val_subset.dataset.transform_rgb = val_transform
    val_subset.dataset.transform_ela = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = MultiStreamCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0
        for rgb, ela, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
            rgb, ela = rgb.to(device), ela.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, ela)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgb.size(0)

        model.eval()
        val_running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for rgb, ela, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
                rgb, ela, labels = rgb.to(device), ela.to(device), labels.to(device)
                outputs = model(rgb, ela)
                v_loss = criterion(outputs, labels.float().unsqueeze(1))
                val_running_loss += v_loss.item() * rgb.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / train_size
        epoch_val_loss = val_running_loss / val_size
        val_acc = (correct / total) * 100

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "multi_stream_best_model.pth")
            print(f"--> Best Multi-Stream model saved at epoch {epoch}")

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch}: Loss: {epoch_loss:.4f} | ValLoss: {epoch_val_loss:.4f} | ValAcc: {val_acc:.2f}%")

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

    plt.title("MULTI-STREAM CNN: Training & Validation Loss", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, color='green', label='Validation Accuracy', linewidth=2)
    plt.title("MULTI-STREAM CNN: Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("multi_stream_plot.png")
    plt.show()