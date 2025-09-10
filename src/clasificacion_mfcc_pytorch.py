import librosa
import numpy as np
import soundfile as sf
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(torch.cuda.is_available())

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=24000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

data = []
labels = []

# Audios reales
real_dir = "Audios_48samplerate/Normal"
fake_dir = "fake_audios"
for file in os.listdir(real_dir):
    if file.endswith(".wav"):
        features = extract_mfcc(os.path.join(real_dir, file))
        data.append(features)
        labels.append(0)  # 0 = real

# Audios falsos
for file in os.listdir(fake_dir):
    if file.endswith(".wav"):
        features = extract_mfcc(os.path.join("fake_audios", file))
        data.append(features)
        labels.append(1)  # 1 = fake

X = np.array(data)
y = np.array(labels)


# Convertir a tensores con tipos correctos
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dividir en train, validación y test
X_temp, X_test, y_temp, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
# 0.6 train / 0.2 val / 0.2 test

# Datasets y dataloaders
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)
test_dl = DataLoader(test_ds, batch_size=8)

# Modelo
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(40):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Validación
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            acc += (pred.argmax(1) == yb).sum().item()
            total += yb.size(0)
    print(f"Epoch {epoch+1:02d} - Validación: Accuracy = {acc/total:.2%}")

# Evaluación en test
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu()
        all_preds.extend(pred.numpy())
        all_labels.extend(yb.numpy())

print("\nResultados en Test:")


print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1],
    target_names=["Real", "Falsa"],
    zero_division=0 
))

print("Matriz de confusión:")
print(confusion_matrix(all_labels, all_preds))