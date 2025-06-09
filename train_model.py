import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.model import GasSensorNN
from src.utils.preprocess import load_and_merge_data, preprocess_data
from src.utils.utils import plot_training_metrics

def train():
    dataset_name = input("Select dataset name: ").strip()
    data, class_names = load_and_merge_data(f"{dataset_name}")
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(data, class_names)
    
    # Konversi ke Tensor
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    # Inisialisasi model
    model = GasSensorNN(X_train.shape[1], len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(50):
        # ... (training loop code)
    
    # Simpan model dan scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, "nn_model.pth")
    
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    train()