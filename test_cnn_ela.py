import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image, ImageChops, ImageEnhance
import io
import multiprocessing

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

class CNN128_ELA(nn.Module):
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_transform = transforms.Compose([
        ELATransform(quality=90, scale=15.0),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    try:
        test_data = datasets.ImageFolder("Cifake_Dataset/test", transform=test_transform)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    model = CNN128_ELA().to(device)
    model_path = "cnn128_ela_best_model.pth" 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded successfully: {model_path}")
    except FileNotFoundError:
        print(f"Error: Δεν βρέθηκε το αρχείο {model_path}.")
        exit()

    model.eval()

    y_true = []
    y_pred = []

    print(f"Ξεκινάει το Testing σε {len(test_data)} εικόνες...")
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            
            preds = (outputs > 0.5).float().cpu().numpy()

            y_true.extend(yb.numpy())
            y_pred.extend(preds.flatten())

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("-" * 30)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ TESTING (ELA-CNN):")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("-" * 30)