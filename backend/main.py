from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import json
from model import FruitClassifier, load_classes

app = FastAPI(title="Fruit Classifier API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = []

@app.on_event("startup")
async def load_model():
    global model, class_names
    
    # Load class names
    if os.path.exists("backend/classes.json"):
        class_names = load_classes("backend/classes.json")
    else:
        print("Warning: classes.json not found, using generic class indices.")
        class_names = [f"Class {i}" for i in range(50)]

    # Initialize and load model
    model = FruitClassifier(num_classes=len(class_names)).to(device)
    model_path = "backend/fruit_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: {model_path} not found. Please train the model first.")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        
        return {
            "prediction": predicted_class,
            "confidence": f"{confidence.item() * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
