import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

dataset_name = input("Select dataset name: ").strip()

# Folder tempat dataset disimpan
dataset_folder = f"{dataset_name}"

# Menggabungkan semua file CSV dalam folder
all_dataframes = []
file_names = []
sample_labels = []  # Untuk menyimpan label setiap sampel

def create_lagged_features(df, num_lags=2):    
    # Buat DataFrame untuk hasil
    result = pd.DataFrame()
    
    for lag in range(0, num_lags + 1):
                    
        # Untuk setiap kolom, buat kolom lag sesuai jumlah yang diminta
        for col in df.columns:
            result[f"{col} (-{lag})"] = df[col].shift(lag)
            
            # Tambahkan kolom asli
            # result[f"{col}"] = df[col]
                
            # Tambahkan kolom lag

    # Hapus baris dengan nilai NaN (baris awal yang tidak punya cukup lag)
    result = result.dropna().reset_index(drop=True)
    
    return result

def clean_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Cari jumlah kolom dari header
    num_columns = len(lines[0].strip().split(','))
    
    # Simpan hanya baris dengan kolom sesuai
    clean_lines = [lines[0]]  # Header
    for line in lines[1:]:
        if len(line.strip().split(',')) == num_columns:
            clean_lines.append(line)
    
    # Simpan ke file sementara
    clean_path = file_path.replace('.csv', '_clean.csv')
    with open(clean_path, 'w') as f:
        f.writelines(clean_lines)
    
    return clean_path

for f in os.listdir(dataset_folder):
    if f.endswith('.csv'):
        filename = os.path.splitext(f)[0]
        file_path = clean_csv(os.path.join(dataset_folder, f))
        
        # Baca file CSV
        df = pd.read_csv(file_path).drop('Timestamp', axis=1)
        # df = create_lagged_features(df,10)
        # Tambah kolom dengan nama file
        df['source_file'] = filename
        
        # Simpan DataFrame dan label
        all_dataframes.append(df)
        file_names.append(filename)
        sample_labels.extend([filename] * len(df))  # Label untuk setiap sampel

# Menggabungkan semua DataFrame menjadi satu
data = pd.concat(all_dataframes, ignore_index=True)

data_final = []

# Dapatkan class_names otomatis dari file_names (hanya nama unik)
class_names = sorted(list(set(file_names)))  # Diurutkan untuk konsistensi

# Fungsi untuk one-hot encoding
def one_hot_encode(labels, class_names):
    one_hot_labels = []
    for name in labels:
        one_hot = [1 if name == class_name else 0 for class_name in class_names]
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels, dtype=np.float32)

# Pisahkan input (sensor) dan output
X = data.drop('source_file', axis=1).values  # Hapus kolom label
y = one_hot_encode(sample_labels, class_names)  # Gunakan sample_labels yang sudah dibuat


# Normalisasi data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler untuk digunakan saat prediksi
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Konversi data ke format tensor PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Buat DataLoader untuk batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definisikan model menggunakan PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.01)  # Regularisasi
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.output_layer(x)  # Tidak pakai softmax di sini
        return x

print(X_train.shape[1])

# Inisialisasi model
model = NeuralNetwork(X_train.shape[1], len(class_names))

# Loss function dan optimizer (gunakan CrossEntropyLoss yang lebih cocok)
criterion = nn.CrossEntropyLoss()  # Untuk klasifikasi multi-kelas
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # CrossEntropyLoss mengharapkan raw logits (tidak di-softmax) dan label class index
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Menghitung akurasi
        _, predicted = torch.max(outputs.data, 1)
        _, actual = torch.max(labels.data, 1)
        correct += (predicted == actual).sum().item()
        total += labels.size(0)
    
    # Simpan loss dan akurasi per epoch
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)
    
    # Evaluasi pada data testing
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            correct += (predicted == actual).sum().item()
            total += labels.size(0)

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct / total)

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, "
              f"Train Acc: {train_accuracies[-1]*100:.2f}%, "
              f"Test Acc: {test_accuracies[-1]*100:.2f}%")

# Evaluasi model akhir
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    _, actual = torch.max(y_test_tensor.data, 1)
    correct = (predicted == actual).sum().item()
    accuracy = correct / len(y_test_tensor) * 100

print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

# Simpan model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'input_size': X_train.shape[1]
}, "nn_model.pth")
print("Model telah disimpan sebagai nn_model.pth")

# Plot grafik
plt.figure(figsize=(15, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
