import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


IMG_SIZE = 128
BATCH_SIZE = 64
FEATURE_DIM = 18
MODEL_PATH = "hybrid_best_model.pth"
CSV_PATH = "features_test_3rd.csv"
IMAGE_ROOT = "Cifake_Dataset/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class HybridTestDataset(Dataset):
    def __init__(self, csv_file, image_root, transform):
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
        subfolder = "REAL" if self.labels[idx] == 0 else "FAKE"
        img_path = f"{self.image_root}/{subfolder}/{self.filenames[idx]}"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, features, label

class HybridCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(FEATURE_DIM, 64),
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

test_dataset = HybridTestDataset(CSV_PATH, IMAGE_ROOT, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = HybridCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, features, labels in test_loader:
        images = images.to(device)
        features = features.to(device)

        outputs = model(images, features)
        preds = (outputs > 0.5).cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\nΑΠΟΤΕΛΕΣΜΑΤΑ TESTING (HYBRID):")
print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
