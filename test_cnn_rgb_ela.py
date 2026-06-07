import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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
        
        if self.transform_rgb:
            img_rgb = self.transform_rgb(img)
        if self.transform_ela:
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
        f_rgb = self.rgb_stream(rgb).flatten(1)
        f_ela = self.ela_stream(ela).flatten(1)
        combined = torch.cat((f_rgb, f_ela), dim=1)
        return self.fc(combined)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    test_data = MultiStreamDataset("Cifake_Dataset/test", test_transform, test_transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    model = MultiStreamCNN().to(device)
    model_path = "multi_stream_best_model.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Weights loaded: {model_path}")
    except FileNotFoundError:
        print("Error: Το αρχείο βαρών δεν βρέθηκε!")
        exit()

    model.eval()
    y_true, y_pred = [], []

    print(f"Ξεκινάει το Testing σε {len(test_data)} εικόνες...")
    with torch.no_grad():
        for rgb, ela, labels in test_loader:
            rgb, ela = rgb.to(device), ela.to(device)
            outputs = model(rgb, ela)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.flatten())

    print("\n" + "="*35)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ TESTING (MULTI-STREAM):")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
    print("="*35)