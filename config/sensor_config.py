SENSOR_CONFIG = {
    "MQ-3": {
        "channel": 0,
        "type": "Alcohol",
        "unit": "V"
    },
    "MQ-135": {
        "channel": 1,
        "type": "Air Pollutants",
        "unit": "V"
    },
    "MQ-136": {
        "channel": 2,
        "type": "H2S",
        "unit": "V"
    },
    "TGS 822": {
        "channel": 7,
        "type": "Organic Vapors",
        "unit": "V"
    },
    "TGS 2600": {
        "channel": 3,
        "type": "General Air Quality",
        "unit": "V"
    },
    "TGS 2602": {
        "channel": 4,
        "type": "Air Pollutants",
        "unit": "V"
    },
    "TGS 2610": {
        "channel": 5,
        "type": "LPG/Propane",
        "unit": "V"
    },
    "TGS 2620": {
        "channel": 6,
        "type": "Alcohol/Volatile Gases",
        "unit": "V"
    }
}

SAMPLING_RATE_HZ = 0.5  # 2 detik per sampel
SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED_HZ = 135000