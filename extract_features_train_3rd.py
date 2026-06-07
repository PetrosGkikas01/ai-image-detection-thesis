import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATASET_PATH_REAL = r"Cifake_Dataset/train/REAL"
DATASET_PATH_FAKE = r"Cifake_Dataset/train/FAKE"
OUTPUT_CSV = "features_train_3rd.csv"
GAMMAS = [0.8, 1.0, 1.2]

data = []

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def RGB_to_RYB(rgb):
    r, g, b = rgb[:, :, 0].copy(), rgb[:, :, 1].copy(), rgb[:, :, 2].copy()

    w = np.minimum(np.minimum(r, g), b)
    r -= w; g -= w; b -= w
    y = np.minimum(r, g)
    r -= y; g -= y
    b += g; y += g
    g = 0
    r += w; y += w; b += w

    ryb = np.stack([r, y, b], axis=-1)
    return np.clip(ryb, 0, 1)

def compute_features(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0

    features = []
    for gamma in GAMMAS:
        corrected = gamma_correction((image * 255).astype(np.uint8), gamma) / 255.0
        corrected = RGB_to_RYB(corrected)

        for c in range(3):  # R, Y, B
            features.append(np.mean(corrected[:, :, c]))
            features.append(np.var(corrected[:, :, c]))
    return features

def process_folder(folder_path, label):
    for filename in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing {folder_path}"):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder_path, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feats = compute_features(image)
        feats.append(filename)        # <-- filename
        feats.append(label)           # <-- label last

        data.append(feats)



process_folder(DATASET_PATH_REAL, 0)
process_folder(DATASET_PATH_FAKE, 1)

columns = [
    "Mean_R(smaller_γ)", "Var_R(smaller_γ)", "Mean_Y(smaller_γ)", "Var_Y(smaller_γ)", "Mean_B(smaller_γ)", "Var_B(smaller_γ)",
    "Mean_R", "Var_R", "Mean_Y", "Var_Y", "Mean_B", "Var_B", "Mean_R(bigger_γ)", "Var_R(bigger_γ)", "Mean_Y(bigger_γ)", "Var_Y(bigger_γ)",
    "Mean_B(bigger_γ)", "Var_B(bigger_γ)",
    "filename",
    "label"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Αποθηκεύτηκε το αρχείο χαρακτηριστικών: {OUTPUT_CSV}")
