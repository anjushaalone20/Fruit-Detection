# Fruit Classifier Project

Build an Image Classifier for Fruits capable of distinguishing 50 fruits using a Convolutional Neural Network (PyTorch) with a beautiful Vanilla frontend. 

## Project Structure

- `backend/`: FastAPI app (`main.py`), PyTorch CNN classes (`model.py`), and the training script (`train.py`)
- `frontend/`: The Web UI (`index.html`, `style.css`, `script.js`)

## Getting Started

### 1. Requirements

Ensure you have Python 3 installed. Install the requirements:
```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model
You must train the model to generate the weights (`fruit_model.pth`) and class labels (`classes.json`).
```bash
python backend/train.py
```
This will run a 3-epoch transfer learning loop using PyTorch MobileNetV2.

### 3. Run the Backend
First load the model, then start the API server:
```bash
python backend/model.py
python backend/main.py
```

### 4. Access the Frontend
Open `frontend/index.html` in any modern web browser. You can now drag and drop fruit images to classify them using your newly trained model!
