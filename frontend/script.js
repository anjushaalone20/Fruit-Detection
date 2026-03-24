const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const loadingSession = document.getElementById('loading');
const resultCard = document.getElementById('result');
const predictedClass = document.getElementById('predicted-class');
const predictedConfidence = document.getElementById('predicted-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const resetBtn = document.getElementById('reset-btn');

// Define API URL
const API_URL = 'http://127.0.0.1:8000/predict';

// Event Listeners for Browse and File Input
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

// Drag and Drop Effects
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => uploadZone.classList.add('drag-active'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('drag-active'), false);
});

uploadZone.addEventListener('drop', (e) => {
    let dt = e.dataTransfer;
    let files = dt.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

// Image Upload Handler
function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file!');
        return;
    }

    // Set preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        
        // Hide upload zone, show preview section
        uploadZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
        
        // Setup UI for loading
        previewImage.style.opacity = '1';
        loadingSession.classList.remove('hidden');
        resultCard.classList.add('hidden');
        resetBtn.style.display = 'none';

        // Call API
        classifyImage(file);
    };
    reader.readAsDataURL(file);
}

// Call FastAPI Backend
async function classifyImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Simulate a tiny delay for smooth animation transition
        setTimeout(() => {
            loadingSession.classList.add('hidden');
            
            if (!response.ok) {
                alert(data.error || 'Failed to classify image.');
                resetUI();
                return;
            }

            displayResult(data.prediction, data.confidence);
        }, 800);

    } catch (error) {
        console.error('Error classifying image:', error);
        loadingSession.classList.add('hidden');
        alert('Could not connect to the server. Is the FastAPI backend running?');
        resetUI();
    }
}

// Display Result
function displayResult(prediction, confidence) {
    const confVal = parseFloat(confidence.replace('%', ''));
    
    // Format presentation
    predictedClass.textContent = prediction.replace('_', ' ').replace(/(^\w|\s\w)/g, m => m.toUpperCase());
    predictedConfidence.textContent = confidence;
    confidenceBar.style.width = `0%`; // Reset width for animation
    
    // Animate UI elements
    resultCard.classList.remove('hidden');
    resultCard.style.animation = 'slideUp 0.5s ease forwards';
    resetBtn.style.display = 'flex';
    resetBtn.style.animation = 'fadeIn 0.5s ease 0.3s forwards';
    
    // Animate progress bar
    setTimeout(() => {
        confidenceBar.style.width = `${confVal}%`;
        
        // Dynamic color based on confidence
        if(confVal >= 80) confidenceBar.style.background = 'linear-gradient(90deg, #10b981 0%, #34d399 100%)';
        else if(confVal >= 50) confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)';
        else confidenceBar.style.background = 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)';
    }, 100);
}

// Reset the UI
function resetUI() {
    previewImage.src = '';
    previewSection.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    fileInput.value = '';
    
    // Reset animations
    resultCard.style.animation = 'none';
    resetBtn.style.animation = 'none';
}

resetBtn.addEventListener('click', resetUI);
