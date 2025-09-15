from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import uuid
import base64
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Class mapping
class_names = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy'
]

# Rich disease information
disease_info = {
    'Anthracnose': {
        "description": "A fungal disease causing dark, sunken lesions on leaves, stems, flowers, and fruits.",
        "symptoms": ["Irregular brown or black leaf spots", "Premature leaf drop", "Fruit rot"],
        "prevention": ["Ensure good air circulation", "Prune dense foliage", "Avoid overhead irrigation"],
        "treatment": ["Apply copper-based fungicides weekly during infection periods"],
        "product": {
            "name": "Bonide Copper Fungicide",
            "link": "https://www.amazon.com/dp/B01M1F2D8Z"
        },
        "more_info": {
            "name": "Anthracnose Treatment Guide",
            "link": "https://www.youtube.com/watch?v=BSXoanCSPic"
        },
        "icon": "fa-solid fa-leaf",
        "color": "#ff5722"
    },
    'Bacterial Canker': {
        "description": "Bacterial disease causing lesions and wilting, leading to reduced yield and quality.",
        "symptoms": ["Water-soaked spots", "Gumming on stems", "Wilted or scorched leaves"],
        "prevention": ["Use sterile pruning tools", "Prune during dry conditions", "Avoid over-fertilization"],
        "treatment": ["Copper-based sprays applied at bud break and during bloom"],
        "product": {
            "name": "Southern Ag Copper Fungicide",
            "link": "https://www.amazon.com/dp/B00HXOI3U6"
        },
        "more_info": {
            "name": "Bacterial Canker Guide",
            "link": "https://www.youtube.com/watch?v=6djij_KJxX0"
        },
        "icon": "fa-solid fa-bacteria",
        "color": "#e91e63"
    },
    'Cutting Weevil': {
        "description": "Pests that chew through young plant tissue and damage growth points.",
        "symptoms": ["Irregular holes in leaves", "Damaged buds and flowers"],
        "prevention": ["Remove weeds and debris around plants", "Avoid late planting"],
        "treatment": ["Use neem oil or biological pesticides like spinosad"],
        "product": {
            "name": "Neem Oil Insecticide",
            "link": "https://www.amazon.com/dp/B004U7GA7A"
        },
        "more_info": {
            "name": "Cutting Weevil Control",
            "link": "https://www.youtube.com/watch?v=GzMoq5QT8EQ"
        },
        "icon": "fa-solid fa-bug",
        "color": "#ff9800"
    },
    'Die Back': {
        "description": "A condition where twigs die back from the tips due to fungal pathogens.",
        "symptoms": ["Wilting tips", "Branch die-off", "Leaf discoloration"],
        "prevention": ["Good drainage", "Avoid waterlogging", "Prune affected branches"],
        "treatment": ["Use systemic fungicides", "Improve air circulation"],
        "product": {
            "name": "Ferti-Lome Systemic Fungicide",
            "link": "https://www.amazon.com/dp/B01MT80ROH"
        },
        "more_info": {
            "name": "How to Handle Die Back",
            "link": "https://www.youtube.com/watch?v=G9lFU6QbZuM"
        },
        "icon": "fa-solid fa-seedling",
        "color": "#795548"
    },
    'Gall Midge': {
        "description": "Tiny flies whose larvae cause abnormal tissue swelling (galls) that hinder plant development.",
        "symptoms": ["Curling leaves", "Stem swellings", "Stunted growth"],
        "prevention": ["Destroy infested shoots early", "Use traps", "Avoid heavy nitrogen fertilizers"],
        "treatment": ["Neem-based or contact insecticides are effective"],
        "product": {
            "name": "Safer Insect Killing Soap",
            "link": "https://www.amazon.com/dp/B00192AO90"
        },
        "more_info": {
            "name": "Gall Midge Guide",
            "link": "https://www.youtube.com/watch?v=GTwV8CtR3iU"
        },
        "icon": "fa-solid fa-mosquito",
        "color": "#9c27b0"
    },
    'Healthy': {
        "description": "Your plant appears healthy and disease-free.",
        "symptoms": ["Vibrant green leaves", "Even growth", "No visible damage"],
        "prevention": ["Regular pruning", "Balanced fertilizer", "Consistent irrigation"],
        "treatment": ["Continue normal care routines"],
        "product": {
            "name": "General Plant Health Kit",
            "link": "https://www.amazon.com/dp/B01M7TG7Q0"
        },
        "more_info": {
            "name": "Healthy Mango Tree Care",
            "link": "https://www.youtube.com/watch?v=KQu_NTKzxoA"
        },
        "icon": "fa-solid fa-check-circle",
        "color": "#4caf50"
    }
}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_class, confidence

# Main HTML template
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlantGuard AI | Plant Disease Detection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <style>
        :root {
            --primary: #4CAF50;
            --primary-light: #80e27e;
            --primary-dark: #087f23;
            --secondary: #ff9800;
            --text-dark: #263238;
            --text-light: #f5f5f5;
            --background: #f9fafb;
            --white: #ffffff;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: var(--background);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        header {
            background-color: var(--white);
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
        }
        
        .logo {
            display: flex;
            align-items: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            text-decoration: none;
        }
        
        .logo i {
            margin-right: 0.5rem;
            font-size: 1.8rem;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
        }
        
        .nav-links li {
            margin-left: 1.5rem;
        }
        
        .nav-links a {
            color: var(--text-dark);
            text-decoration: none;
            font-weight: 500;
            transition: var(--transition);
        }
        
        .nav-links a:hover {
            color: var(--primary);
        }
        
        .mobile-menu {
            display: none;
            font-size: 1.5rem;
            cursor: pointer;
        }
        
        .hero {
            padding: 4rem 0;
            background: linear-gradient(135deg, var(--primary-light), var(--primary));
            color: var(--text-light);
            text-align: center;
            border-radius: 0 0 20px 20px;
            margin-bottom: 3rem;
        }
        
        .hero h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            animation: fadeInDown 1s;
        }
        
        .hero p {
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0 auto 2rem;
            animation: fadeIn 1s 0.5s both;
        }
        
        .main-content {
            padding: 2rem 0;
        }
        
        .upload-container {
            background-color: var(--white);
            border-radius: 10px;
            box-shadow: var(--shadow);
            padding: 2rem;
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInUp 1s;
        }
        
        .upload-area {
            border: 2px dashed var(--primary-light);
            border-radius: 10px;
            padding: 3rem 1rem;
            margin-bottom: 1.5rem;
            position: relative;
            cursor: pointer;
            transition: var(--transition);
        }
        
        .upload-area:hover {
            border-color: var(--primary);
            background-color: rgba(76, 175, 80, 0.05);
        }
        
        .upload-area i {
            font-size: 3rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }
        
        .upload-area h3 {
            margin-bottom: 0.5rem;
        }
        
        .btn {
            display: inline-block;
            background-color: var(--primary);
            color: var(--white);
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            text-decoration: none;
            transition: var(--transition);
            box-shadow: var(--shadow);
        }
        
        .btn:hover {
            background-color: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background-color: var(--secondary);
        }
        
        .btn-secondary:hover {
            background-color: #f57c00;
        }
        
        #file-input {
            display: none;
        }
        
        #upload-btn {
            display: none;
            margin-top: 1rem;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }
        
        .feature-card {
            background-color: var(--white);
            border-radius: 10px;
            box-shadow: var(--shadow);
            padding: 1.5rem;
            text-align: center;
            transition: var(--transition);
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .feature-card i {
            font-size: 2.5rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }
        
        .feature-card h3 {
            margin-bottom: 0.5rem;
        }
        
        footer {
            background-color: var(--text-dark);
            color: var(--text-light);
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
        }
        
        .social-links {
            display: flex;
            justify-content: center;
            list-style: none;
            margin: 1rem 0;
        }
        
        .social-links li {
            margin: 0 0.5rem;
        }
        
        .social-links a {
            color: var(--text-light);
            font-size: 1.5rem;
            transition: var(--transition);
        }
        
        .social-links a:hover {
            color: var(--primary-light);
        }
        
        /* Result Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            overflow-y: auto;
        }
        
        .modal-content {
            background-color: var(--white);
            border-radius: 10px;
            width: 90%;
            max-width: 800px;
            margin: 5% auto;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 0.5s;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .modal-header h2 {
            display: flex;
            align-items: center;
        }
        
        .modal-header i {
            margin-right: 0.5rem;
            font-size: 1.5rem;
        }
        
        .close-modal {
            font-size: 1.5rem;
            cursor: pointer;
            color: #888;
            transition: var(--transition);
        }
        
        .close-modal:hover {
            color: var(--text-dark);
        }
        
        .modal-body {
            padding: 1.5rem;
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        .result-image {
            width: 100%;
            border-radius: 10px;
            box-shadow: var(--shadow);
        }
        
        .result-info {
            display: flex;
            flex-direction: column;
        }
        
        .diagnosis {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 10px;
            color: var(--white);
        }
        
        .diagnosis i {
            font-size: 2rem;
            margin-right: 1rem;
        }
        
        .confidence-bar {
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-bottom: 2rem;
            overflow: hidden;
        }
        
        .confidence-value {
            height: 100%;
            background-color: var(--primary);
            border-radius: 5px;
            transition: width 1s ease-in-out;
        }
        
        .info-section {
            margin-bottom: 1.5rem;
        }
        
        .info-section h3 {
            margin-bottom: 0.5rem;
            color: var(--text-dark);
            display: flex;
            align-items: center;
        }
        
        .info-section h3 i {
            margin-right: 0.5rem;
        }
        
        .info-section ul {
            list-style-type: none;
            padding-left: 1.5rem;
        }
        
        .info-section ul li {
            margin-bottom: 0.5rem;
            position: relative;
        }
        
        .info-section ul li:before {
            content: "•";
            color: var(--primary);
            font-weight: bold;
            position: absolute;
            left: -1rem;
        }
        
        .action-links {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            margin: 1rem auto;
            border-radius: 5px;
            display: none;
        }
        
        .loader {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .loader i {
            font-size: 3rem;
            color: var(--primary);
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Media Queries */
        @media (max-width: 768px) {
            .nav-links {
                display: none;
            }
            
            .mobile-menu {
                display: block;
            }
            
            .hero h1 {
                font-size: 2rem;
            }
            
            .result-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <a href="/" class="logo">
                    <i class="fas fa-leaf"></i>
                    PlantGuard AI
                </a>
                <ul class="nav-links">
                    <li><a href="#"><i class="fas fa-home"></i> Home</a></li>
                    <li><a href="#"><i class="fas fa-history"></i> History</a></li>
                    <li><a href="#"><i class="fas fa-book"></i> Guide</a></li>
                    <li><a href="#"><i class="fas fa-info-circle"></i> About</a></li>
                </ul>
                <div class="mobile-menu">
                    <i class="fas fa-bars"></i>
                </div>
            </nav>
        </div>
    </header>

    <section class="hero">
        <div class="container">
            <h1>Detect Plant Diseases Instantly</h1>
            <p>Upload a photo of your plant and our AI will identify diseases with precision, providing actionable treatment recommendations.</p>
        </div>
    </section>

    <section class="main-content">
        <div class="container">
            <div class="upload-container">
                <h2>Upload Plant Image</h2>
                <p>Take a clear photo of the affected area for the most accurate results</p>
                
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="upload-area" id="drop-area">
                        <i class="fas fa-cloud-upload-alt"></i>
                        <h3>Drag & Drop</h3>
                        <p>or click to browse files</p>
                        <input type="file" id="file-input" name="file" accept="image/*">
                    </div>
                    <img id="preview-image" class="preview-image" alt="Preview">
                    <button type="submit" id="upload-btn" class="btn">Analyze Image</button>
                </form>
                
                <div class="loader" id="loader">
                    <i class="fas fa-spinner"></i>
                    <p>Analyzing plant image...</p>
                </div>
            </div>

            <div class="features">
                <div class="feature-card">
                    <i class="fas fa-bolt"></i>
                    <h3>Fast Results</h3>
                    <p>Get instant disease detection with our advanced AI model.</p>
                </div>
                <div class="feature-card">
                    <i class="fas fa-list-check"></i>
                    <h3>Detailed Reports</h3>
                    <p>Receive comprehensive information about detected diseases.</p>
                </div>
                <div class="feature-card">
                    <i class="fas fa-flask"></i>
                    <h3>Treatment Plans</h3>
                    <p>Access expert-backed treatment recommendations.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Result Modal -->
    <div class="modal" id="result-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2><i id="result-icon" class="fas fa-leaf"></i> <span id="result-title">Disease Detection</span></h2>
                <span class="close-modal" id="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <div class="result-grid">
                    <div>
                        <img id="result-image" class="result-image" alt="Plant Image">
                    </div>
                    <div class="result-info">
                        <div id="diagnosis" class="diagnosis">
                            <i id="diagnosis-icon" class="fas fa-check-circle"></i>
                            <div>
                                <h3 id="disease-name">Healthy</h3>
                                <p id="confidence">Confidence: 95%</p>
                            </div>
                        </div>
                        
                        <div class="confidence-bar">
                            <div id="confidence-value" class="confidence-value" style="width: 95%;"></div>
                        </div>
                        
                        <div class="info-section">
                            <h3><i class="fas fa-info-circle"></i> Description</h3>
                            <p id="disease-description">Your plant appears healthy and disease-free.</p>
                        </div>
                        
                        <div class="info-section">
                            <h3><i class="fas fa-exclamation-triangle"></i> Symptoms</h3>
                            <ul id="symptoms-list"></ul>
                        </div>
                        
                        <div class="info-section">
                            <h3><i class="fas fa-shield-alt"></i> Prevention</h3>
                            <ul id="prevention-list"></ul>
                        </div>
                        
                        <div class="info-section">
                            <h3><i class="fas fa-first-aid"></i> Treatment</h3>
                            <ul id="treatment-list"></ul>
                        </div>
                        
                        <div class="action-links">
                            <a id="product-link" href="#" class="btn" target="_blank">
                                <i class="fas fa-shopping-cart"></i> Recommended Product
                            </a>
                            <a id="info-link" href="#" class="btn btn-secondary" target="_blank">
                                <i class="fas fa-video"></i> Watch Guide
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="container">
            <p>&copy; 2025 PlantGuard AI. All rights reserved.</p>
            <ul class="social-links">
                <li><a href="#"><i class="fab fa-facebook"></i></a></li>
                <li><a href="#"><i class="fab fa-twitter"></i></a></li>
                <li><a href="#"><i class="fab fa-instagram"></i></a></li>
                <li><a href="#"><i class="fab fa-youtube"></i></a></li>
            </ul>
            <p>Powered by TensorFlow | Built for plant enthusiasts</p>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('file-input');
            const dropArea = document.getElementById('drop-area');
            const uploadForm = document.getElementById('upload-form');
            const previewImage = document.getElementById('preview-image');
            const uploadBtn = document.getElementById('upload-btn');
            const loader = document.getElementById('loader');
            const resultModal = document.getElementById('result-modal');
            const closeModal = document.getElementById('close-modal');
            
            // Preview image
            fileInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                        previewImage.style.display = 'block';
                        uploadBtn.style.display = 'inline-block';
                    }
                    reader.readAsDataURL(file);
                }
            });
            
            // Drag and drop
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropArea.style.borderColor = 'var(--primary)';
                dropArea.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
            }
            
            function unhighlight() {
                dropArea.style.borderColor = 'var(--primary-light)';
                dropArea.style.backgroundColor = '';
            }
            
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const file = dt.files[0];
                
                if (file && file.type.startsWith('image/')) {
                    fileInput.files = dt.files;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                        previewImage.style.display = 'block';
                        uploadBtn.style.display = 'inline-block';
                    }
                    reader.readAsDataURL(file);
                }
            }
            
            // Click to select file
            dropArea.addEventListener('click', function() {
                fileInput.click();
            });
            
            // Form submission
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                
                if (!fileInput.files[0]) {
                    alert('Please select an image first.');
                    return;
                }
                
                // Show loader
                uploadBtn.style.display = 'none';
                loader.style.display = 'block';
                
                // Send request
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayResults(data);
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing your request.');
                })
                .finally(() => {
                    loader.style.display = 'none';
                    uploadBtn.style.display = 'inline-block';
                });
            });
            
            // Display results
            function displayResults(data) {
                // Set image
                document.getElementById('result-image').src = '/' + data.image_path;
                
                // Set disease info
                document.getElementById('result-title').textContent = data.disease + ' Detected';
                document.getElementById('disease-name').textContent = data.disease;
                document.getElementById('confidence').textContent = 'Confidence: ' + data.confidence.toFixed(1) + '%';
                document.getElementById('confidence-value').style.width = data.confidence + '%';
                document.getElementById('disease-description').textContent = data.info.description;
                
                // Set icon
                const resultIcon = document.getElementById('result-icon');
                const diagnosisIcon = document.getElementById('diagnosis-icon');
                resultIcon.className = data.info.icon;
                diagnosisIcon.className = data.info.icon;
                
                // Set diagnosis color
                const diagnosis = document.getElementById('diagnosis');
                diagnosis.style.backgroundColor = data.info.color;
                
                // Set lists
                setList('symptoms-list', data.info.symptoms);
                setList('prevention-list', data.info.prevention);
                setList('treatment-list', data.info.treatment);
                
                // Set links
                document.getElementById('product-link').href = data.info.product.link;
                document.getElementById('product-link').textContent = '🛒 ' + data.info.product.name;
                
                document.getElementById('info-link').href = data.info.more_info.link;
                document.getElementById('info-link').textContent = '📹 ' + data.info.more_info.name;
                
                // Show modal
                resultModal.style.display = 'block';
            }
            
            // Set list items
            function setList(elementId, items) {
                const list = document.getElementById(elementId);
                list.innerHTML = '';
                
                items.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    list.appendChild(li);
                });
            }
            
            // Close modal
            closeModal.addEventListener('click', function() {
                resultModal.style.display = 'none';
            });
            
            // Close modal when clicking outside
            window.addEventListener('click', function(e) {
                if (e.target == resultModal) {
                    resultModal.style.display = 'none';
                }
            });
            
            // Mobile menu toggle
            const mobileMenu = document.querySelector('.mobile-menu');
            const navLinks = document.querySelector('.nav-links');
            
            mobileMenu.addEventListener('click', function() {
                navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
            });
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    # Make prediction
    disease, confidence = predict_disease(file_path)
    
    return jsonify({
        "success": True,
        "image_path": file_path,
        "disease": disease,
        "confidence": confidence,
        "info": disease_info[disease]
    })

if __name__ == "__main__":
    app.run(debug=False, port=700)
