import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

# Uyarıları kapat (isteğe bağlı)
warnings.filterwarnings("ignore")

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yolu
data_dir = r"C:/Users/genca/Desktop/Vehicle Class"

# Görüntü dönüştürme işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasetler
train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_data   = ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
test_data  = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# DataLoader'lar
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ Modeli yükle - UYARISIZ
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_data.classes))
model = model.to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Eğitim fonksiyonu
def train_model(model, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
    print("✅ Eğitim tamamlandı.")

# Test fonksiyonu
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Doğruluk: {acc * 100:.2f}%")
    return all_preds, all_labels

# Görselleştirme fonksiyonu
def show_misclassified_images(preds, labels, dataset):
    wrong = np.where(np.array(preds) != np.array(labels))[0]
    if len(wrong) == 0:
        print("💯 Tüm testler doğru sınıflandırılmış.")
        return
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(wrong[:5]):
        img, true_label = dataset[idx]
        plt.subplot(1, 5, i+1)
        plt.imshow(np.clip(np.transpose(img.numpy(), (1, 2, 0)), 0, 1))
        plt.title(f'Gerçek: {dataset.classes[labels[idx]]}\nTahmin: {dataset.classes[preds[idx]]}')
        plt.axis('off')
    plt.show()

# ▶️ Eğitimi başlat
train_model(model, epochs=10)

# 🧪 Test et
print("🔍 Test Verisi ile Değerlendirme:")
preds, labels = evaluate_model(model, test_loader)

# 💾 MODELİ KAYDET
model_path = "resnet34_vehicle_classifier.pth"
torch.save(model.state_dict(), model_path)
if os.path.exists(model_path):
    print(f"📦 Model başarıyla kaydedildi: {model_path}")
else:
    print("❌ Model kaydedilemedi!")

# 🎨 Yanlış tahmin görselleri
show_misclassified_images(preds, labels, test_data)
