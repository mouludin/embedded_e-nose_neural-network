import matplotlib.pyplot as plt

def plot_training_metrics(train_loss, test_loss, train_acc, test_acc):
    """Plot loss dan accuracy"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_lagged_features(df, num_lags=2):
    """Buat fitur lag (opsional)"""
    result = pd.DataFrame()
    for lag in range(num_lags + 1):
        for col in df.columns:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    return result.dropna()