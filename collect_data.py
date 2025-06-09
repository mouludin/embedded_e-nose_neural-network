from src.data_collection.collector import GasSensorCollector
from src.data_collection.data_handler import DataHandler
from config import sensor_config as cfg
import time

def main():
    try:
        # User input
        sample_name = input("Input sample name/category: ").strip()
        dataset_name = input("Select dataset name: ").strip()
        
        # Inisialisasi modul
        collector = GasSensorCollector()
        data = []
        
        # Siapkan header CSV
        header = ["Timestamp"] + list(cfg.SENSOR_CONFIG.keys())
        filepath = DataHandler.prepare_filepath(dataset_name, sample_name)
        
        print(f"\nRecording data for '{sample_name}'...")
        print("Press Ctrl+C to stop and save\n")
        
        try:
            while True:
                timestamp, readings = collector.read_all_sensors()
                row = collector.format_for_csv(timestamp, readings)
                data.append(row)
                
                # Tampilkan output real-time
                print(f"\n{timestamp}:")
                for sensor, values in readings.items():
                    print(f"{sensor}: {values['value']:.3f}{values['unit']}", end=" | ")
                
                time.sleep(1 / cfg.SAMPLING_RATE_HZ)
                
        except KeyboardInterrupt:
            DataHandler.save_to_csv(filepath, data, header)
            print(f"\nSaved {len(data)} samples to {filepath}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        collector.spi.close()

if __name__ == "__main__":
    main()