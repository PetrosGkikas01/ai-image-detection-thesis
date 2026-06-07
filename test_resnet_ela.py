import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
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

def get_resnet18_ela_model():
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        ELATransform(quality=90),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder("Cifake_Dataset/test", transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = get_resnet18_ela_model().to(device)
    model.load_state_dict(torch.load("resnet18_ela_best_model.pth", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    print(f"Ξεκινάει το Testing ResNet-ELA σε {len(test_data)} εικόνες...")

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = (outputs > 0.5).float().cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds.flatten())

    print("-" * 30)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ TESTING (ResNet18-ELA):")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
    print("-" * 30)