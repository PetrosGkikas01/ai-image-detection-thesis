import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
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

def get_resnet18_ela_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

if __name__ == "__main__":   
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True

    
    IMG_SIZE = 128
    BATCH_SIZE = 64  
    LR = 0.0001
    EPOCHS = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        ELATransform(quality=90),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        ELATransform(quality=90),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder("Cifake_Dataset/train")
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = get_resnet18_ela_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
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
            for xb, yb in val_loader:
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
        val_acc = (correct / total) * 100

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "resnet18_ela_best_model.pth")
            print(f"--> Best ResNet-ELA model saved")

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

    plt.title("ELA RESNET: Training & Validation Loss", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, color='green', label='Validation Accuracy', linewidth=2)
    plt.title("ELA RESNET: Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    plt.savefig("ResNet_plot.png")
    plt.show()
    print("\nΕκπαίδευση ResNet-ELA ολοκληρώθηκε!")