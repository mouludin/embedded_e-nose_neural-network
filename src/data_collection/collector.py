import time
from ..utils.spi_utils import SPIHandler
from config import sensor_config as cfg

class GasSensorCollector:
    def __init__(self):
        self.spi = SPIHandler(cfg.SPI_BUS, cfg.SPI_DEVICE, cfg.SPI_SPEED_HZ)
        
    def read_all_sensors(self):
        """Membaca semua sensor yang terkonfigurasi"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        readings = {
            sensor: {
                "value": self.spi.read_adc(config["channel"]),
                "type": config["type"],
                "unit": config["unit"]
            } for sensor, config in cfg.SENSOR_CONFIG.items()
        }
        return timestamp, readings
    
    def format_for_csv(self, timestamp, readings):
        """Format data untuk penyimpanan CSV"""
        return [timestamp] + [readings[sensor]["value"] for sensor in cfg.SENSOR_CONFIG]