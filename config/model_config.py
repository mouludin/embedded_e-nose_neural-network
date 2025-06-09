MODEL_PATHS = {
    "model": "models/nn_model.pth",
    "scaler": "models/scaler.pkl",
    "min_confidence": 0.7  # Threshold confidence
}

SENSOR_ORDER = [  # Harus sama dengan urutan training
    "MQ-3", "MQ-135", "MQ-136",
    "TGS 2600", "TGS 2602", "TGS 2610",
    "TGS 2620", "TGS 822"
]