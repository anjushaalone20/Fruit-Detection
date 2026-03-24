# Fruit Classifier Project

Build an Image Classifier for Fruits capable of distinguishing 50 fruits using a Convolutional Neural Network (PyTorch) with a beautiful Vanilla frontend. 

## Project Structure

```
Fruit-Detection/
├── backend/
│   ├── main.py            # FastAPI app & /predict endpoint
│   ├── model.py           # FruitClassifier class (MobileNetV2)
│   ├── train.py           # Training script
│   ├── classes.json       # 50 class labels (generated after training)
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── index.html         # Main UI
│   ├── style.css          # Styling & animations
│   └── script.js          # Drag-and-drop, API calls
├── dataset/               # Place downloaded dataset here (not included)
├── .gitignore
└── README.md
```

## Getting Started

### 1. Requirements

Ensure you have Python 3 installed. Install the requirements:
```bash
pip install -r backend/requirements.txt
```

### 2. Download Dataset
This project uses the **Fruits 360** dataset (50 fruit/vegetable classes).

📦 **Download here:** [Fruits 360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)

After downloading, extract and place the data so the structure looks like:
```
dataset/
├── train/
│   ├── Apple Golden/
│   ├── Banana/
│   └── ...
└── val/
    ├── Apple Golden/
    ├── Banana/
    └── ...
```

### 3. Train the Model
You must train the model to generate the weights (`fruit_model.pth`) and class labels (`classes.json`).
```bash
python backend/train.py
```
This will run a transfer learning loop using PyTorch MobileNetV2.

### 4. Run the Backend
First load the model, then start the API server:
```bash
python backend/model.py
python backend/main.py
```

### 5. Access the Frontend
Open `frontend/index.html` in any modern web browser. You can now drag and drop fruit images to classify them using your newly trained model!
