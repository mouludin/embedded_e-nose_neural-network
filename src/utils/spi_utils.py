import spidev

class SPIHandler:
    def __init__(self, bus, device, max_speed_hz):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz
    
    def read_adc(self, channel):
        """Membaca nilai ADC dari channel tertentu"""
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        return ((adc[1] & 3) << 8) + adc[2] * 3.3 / 1023
    
    def close(self):
        self.spi.close()