import torch
import numpy as np
from ..model import GasSensorNN
from config.model_config import MODEL_PATHS

class GasPredictor:
    def __init__(self):
        self.model, self.scaler, self.class_names = self._load_artifacts()
        self.model.eval()
    
    def _load_artifacts(self):
        """Load model dan preprocessing artifacts"""
        model_data = torch.load(MODEL_PATHS["model"])
        scaler = pickle.load(open(MODEL_PATHS["scaler"], "rb"))
        
        model = GasSensorNN(
            input_size=model_data['input_size'],
            output_size=len(model_data['class_names'])
        )
        model.load_state_dict(model_data['model_state_dict'])
        
        return model, scaler, model_data['class_names']
    
    def predict(self, sensor_data):
        """Lakukan prediksi pada data sensor"""
        scaled_data = self.scaler.transform(sensor_data)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            pred_class_idx = np.argmax(probs)
            
        return {
            "class": self.class_names[pred_class_idx],
            "confidence": probs[pred_class_idx],
            "probabilities": dict(zip(self.class_names, probs))
        }