from src.inference.predictor import GasPredictor
from src.inference.sensor_reader import SensorReader
from config.model_config import MODEL_PATHS
import time

def display_results(sensor_readings, prediction):
    """Tampilkan hasil dengan formatting rapi"""
    print("\n" + "="*50)
    print("Sensor Readings (Voltage):")
    for name, value in zip(SENSOR_ORDER, sensor_readings[0]):
        print(f"- {name}: {value:.3f}V")
    
    print("\nPrediction Results:")
    print(f"- Predicted Class: {prediction['class']}")
    print(f"- Confidence: {prediction['confidence']*100:.2f}%")
    
    print("\nClass Probabilities:")
    for cls, prob in prediction['probabilities'].items():
        print(f"- {cls}: {prob*100:.2f}%")
    print("="*50)

def main():
    predictor = GasPredictor()
    reader = SensorReader()
    
    try:
        while True:
            # Baca sensor
            sensor_data = reader.read_sensors()
            
            # Prediksi
            prediction = predictor.predict(sensor_data)
            
            # Tampilkan hasil
            display_results(sensor_data, prediction)
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nPrediksi dihentikan oleh pengguna")
    finally:
        reader.close()

if __name__ == "__main__":
    main()