import time
import spidev
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load model dan scaler
def load_model():
    model_data = torch.load("nn_model.pth")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.layer1 = torch.nn.Linear(input_size, 512)
            self.layer2 = torch.nn.Linear(512, 256)
            self.layer3 = torch.nn.Linear(256, 128)
            self.output_layer = torch.nn.Linear(128, output_size)
            self.dropout = torch.nn.Dropout(0.01)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.dropout(x)
            x = torch.relu(self.layer2(x))
            x = self.dropout(x)
            x = torch.relu(self.layer3(x))
            x = self.output_layer(x)
            return x
    
    model = NeuralNetwork(model_data['input_size'], len(model_data['class_names']))
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    return model, scaler, model_data['class_names']

# [Tambahkan setelah load_model()]
# Pastikan urutan sensor sama dengan training
SENSOR_ORDER = [
    "MQ-3", "MQ-135", "MQ-136",
    "TGS 2600", "TGS 2602", "TGS 2610",
    "TGS 2620", "TGS 822"
]

def get_sensor_readings(spi):
    readings = {sensor: read_adc(pin) for sensor, pin in SENSOR_PINS.items()}
    return np.array([[readings[sensor] for sensor in SENSOR_ORDER]])

# Inisialisasi SPI untuk membaca sensor
def init_spi():
    spi = spidev.SpiDev()
    spi.open(0, 0)  # Bus 0, Device 0
    spi.max_speed_hz = 135000
    return spi

# Konfigurasi pin sensor
SENSOR_PINS = {
    "MQ-3": 0,      # Alkohol
    "MQ-135": 1,    # Polutan Udara
    "MQ-136": 2,    # H2S
    "TGS 822": 7,   # Uap Organik
    "TGS 2600": 3,  # Gas Umum
    "TGS 2602": 4,  # Gas Polutan
    "TGS 2610": 5,  # LPG
    "TGS 2620": 6   # Alkohol dan gas volatil
}

def read_adc(spi, channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    voltage = (data * 3.3) / 1023
    return voltage

def get_sensor_readings(spi):
    sensor_values = []
    for pin in SENSOR_PINS.values():
        sensor_values.append(read_adc(spi, pin))
    return np.array([sensor_values])

def main():
    # Load model dan scaler
    model, scaler, class_names = load_model()
    
    # Inisialisasi SPI
    spi = init_spi()
    
 
    
    try:
        while True:
            # Baca data sensor
            sensor_data = get_sensor_readings(spi)

            scaled_data = scaler.transform(sensor_data)
        
            # Konversi ke tensor
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
                
            # Prediksi
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_class = class_names[predicted.item()]
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                
            # Tampilkan hasil
            print("\n" + "="*50)
            print("Sensor Reading Results:")
            for name, value in zip(SENSOR_PINS.keys(), sensor_data[0]):
                print(f"{name}: {value:.3f} V")
                
            print("\nPrediction Results:")
            print(f"Prediction Class: {predicted_class}")
            print("Probability:")
            for class_name, prob in zip(class_names, probabilities):
                print(f"{class_name}: {prob*100:.2f}%")
                
            print("="*50 + "\n")
            time.sleep(2)  # Jeda 2 detik antar pembacaan
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        spi.close()

if __name__ == "__main__":
    main()
