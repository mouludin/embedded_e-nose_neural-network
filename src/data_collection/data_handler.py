import csv
import os
from datetime import datetime

class DataHandler:
    @staticmethod
    def save_to_csv(filename, data, header):
        """Menyimpan data ke CSV dengan penanganan file yang aman"""
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(data)
    
    @staticmethod
    def prepare_filepath(dataset_name, sample_name):
        """Membuat path file dengan timestamp"""
        os.makedirs(dataset_name, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{dataset_name}/{sample_name}_{timestamp}.csv"