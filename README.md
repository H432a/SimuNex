# SimuNex - AI-Powered IoT Simulation Platform

## Overview
SimuNex is an AI-driven IoT simulation platform that helps users identify components, receive project suggestions, and test hardware simulations in real-time. The platform integrates **YOLOv8** for component detection, **Llama3-70B** for AI assistance, and **HardCode**, a LeetCode-like system for coding and validating IoT projects.

## Features
- **IoT Component Identification**: Uses **YOLOv8** to detect and recognize IoT components with **85% accuracy**.
- **AI-Driven Assistance**: Implements **Llama3-70B** to provide real-time guidance for IoT simulations.
- **HardCode - IoT Coding Platform**: Allows users to simulate hardware connections, write code, and test solutions.

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: YOLOv8, Llama3-70B
- **Database**: SQLite3

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Flask
- OpenCV
- PyTorch (for YOLOv8)
- Llama3-70B Model (API or local deployment)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/Harinee2501/SimuNex.git
   cd SimuNex
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```
4. Open the browser and visit:
   ```sh
   http://localhost:5000
   ```

## Usage
- Upload an image of an IoT component for identification.
- Get AI-generated suggestions for IoT projects.
- Use HardCode to simulate and validate circuit connections.
