from ..utils.spi_utils import SPIHandler
from config import sensor_config as cfg
from config.model_config import SENSOR_ORDER
import numpy as np

class SensorReader:
    def __init__(self):
        self.spi = SPIHandler(cfg.SPI_BUS, cfg.SPI_DEVICE, cfg.SPI_SPEED_HZ)
    
    def read_sensors(self):
        """Baca semua sensor sesuai urutan training"""
        readings = np.array([
            self.spi.read_adc(cfg.SENSOR_CONFIG[sensor]["channel"])
            for sensor in SENSOR_ORDER
        ])
        return readings.reshape(1, -1)  # Reshape untuk single sample
    
    def close(self):
        self.spi.close()