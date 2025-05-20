import time
import spidev
import csv
import os

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 135000

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

def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return (data * 3.3) / 1023

def save_to_csv(filename, data, header):
    """Menyimpan data ke CSV, append jika file sudah ada"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:  # Mode 'a' untuk append
        writer = csv.writer(f)
        
        # Tulis header hanya jika file baru
        if not file_exists:
            writer.writerow(header)
        
        writer.writerows(data)

try:
    sample_name = input("Input sample name/category: ").strip()
    dataset_name = input("Select dataset name: ").strip()
    data = []
    os.makedirs(f"{dataset_name}", exist_ok=True)
    filename = f"{dataset_name}/{sample_name}.csv"
    
    print(f"\nRecording data for '{sample_name}'...")
    print("Press Ctrl+C to stop and save\n")
    
    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            readings = {sensor: read_adc(pin) for sensor, pin in SENSOR_PINS.items()}
            
            row = [timestamp] + [readings[sensor] for sensor in SENSOR_PINS]
            data.append(row)
            
            # Tampilkan pembacaan terbaru
            print(f"\nLast reading at {timestamp}:")
            for sensor, value in readings.items():
                print(f"{sensor}: {value:.3f}V", end=" | ")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        # Simpan data
        header = ["Timestamp"] + list(SENSOR_PINS.keys())
        save_to_csv(filename, data, header)
        
        print(f"\n\nSuccessfully saved {len(data)} records to {filename}")
        if os.path.isfile(filename):
            total_records = sum(1 for _ in open(filename)) - 1  # Kurangi header
            print(f"Total records in file: {total_records}")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
finally:
    spi.close()
    print("SPI connection closed.")
