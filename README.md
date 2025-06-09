# Electronic Nose AI Project 🍃👃

**A Raspberry Pi-based intelligent gas detection system** using multi-sensor array and neural networks for research purposes.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Experimental-orange)](https://your-university.edu)

## 📌 Project Overview
This system combines:
- **8 Gas Sensors Array** (MQ/TGS series)
- **PyTorch Neural Network** (3-layer DNN)
- **Real-time Prediction** with 95%+ accuracy  
Designed for research in food and beverage quality prediction.

## 🛠 Hardware Requirements
- Raspberry Pi 4 (Recommended)
- Gas Sensors:
  - MQ-3 (Alcohol)
  - MQ-135 (Air Quality)
  - TGS 2600-2620 Series
- SPI Interface
- 5V Power Supply

## Sensor circuit on Raspberry Pi

<img src="images/Circuit.png" alt="circuit">

## 🚀 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/mouludin/embedded_e-nose_neural-network.git
cd embedded_e-nose_neural-network

# Install dependencies
pip install -r requirements.txt
