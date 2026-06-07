import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATASET_PATH_TEST_REAL = r"Cifake_Dataset/test/REAL"
DATASET_PATH_TEST_FAKE = r"Cifake_Dataset/test/FAKE"
OUTPUT_CSV = "features_test.csv"


GAMMAS = [0.8, 1.0, 1.2]


data = []

def gamma_correction(image, gamma):
    invertedGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invertedGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def RGB_to_RYB(rgb):
    """Μετατροπή RGB σε RYB (Red-Yellow-Blue χρωματικό χώρο)"""
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
            mean_val = np.mean(corrected[:, :, c])
            var_val = np.var(corrected[:, :, c])
            features.append(mean_val)
            features.append(var_val)
    return features

def process_folder(folder_path, label):
    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_path}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            feats = compute_features(image)
            feats.append(label)  # 0 = REAL, 1 = AI-generated
            data.append(feats)

process_folder(DATASET_PATH_TEST_REAL, 0)
process_folder(DATASET_PATH_TEST_FAKE, 1)

columns = [
    "Mean_R(smaller_γ)", "Var_R(smaller_γ)", "Mean_Y(smaller_γ)", "Var_Y(smaller_γ)", "Mean_B(smaller_γ)", "Var_B(smaller_γ)",
    "Mean_R", "Var_R", "Mean_Y", "Var_Y", "Mean_B", "Var_B",
    "Mean_R(bigger_γ)", "Var_R(bigger_γ)", "Mean_Y(bigger_γ)", "Var_Y(bigger_γ)", "Mean_B(bigger_γ)", "Var_B(bigger_γ)",
    "is_AI_generated"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Αποθηκεύτηκε το αρχείο χαρακτηριστικών TEST: {OUTPUT_CSV}")
