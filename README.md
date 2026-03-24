# Fruit Classifier Project

Build an Image Classifier for Fruits capable of distinguishing 50 fruits using a Convolutional Neural Network (PyTorch) with a beautiful Vanilla frontend. 

## Project Structure

- `backend/`: FastAPI app (`main.py`), PyTorch CNN classes (`model.py`), and the training script (`train.py`)
- `frontend/`: The Web UI (`index.html`, `style.css`, `script.js`) 
- `scripts/`: Tooling for setting up the dataset (`download_dataset.py`)

## Getting Started

### 1. Requirements

Ensure you have Python 3 installed. Install the requirements:
```bash
pip install -r backend/requirements.txt
```

### 2. Download Dataset
This project requires a dataset of 50 fruit classes. Run the download script to automatically fetch and arrange the images.
```bash
python scripts/download_dataset.py
```
*Note: This downloads a 700MB archive and extracts exactly 50 selected classes to save space and training time.*

### 3. Train the Model
You must train the model to generate the weights (`fruit_model.pth`) and class labels (`classes.json`).
```bash
python backend/train.py
```
This will run a 3-epoch transfer learning loop using PyTorch MobileNetV2.

### 4. Run the API Server
Start the backend inference server with FastAPI:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Frontend
Open `frontend/index.html` in any modern web browser. You can now drag and drop fruit images to classify them using your newly trained model!
