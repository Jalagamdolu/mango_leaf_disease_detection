🍃 Mango Leaf Disease Detection using Deep Learning

This project focuses on detecting mango leaf diseases using deep learning techniques. It uses an Inception-based CNN model to classify mango leaf images into healthy or diseased categories. The goal is to support farmers and researchers by providing an automated solution for early disease detection.

📌 Features

Image classification for mango leaf diseases

Deep learning model using Inception architecture

Dataset preprocessing and augmentation

Model evaluation with accuracy and loss visualization

Easy-to-use Jupyter Notebook (training_model_inception.ipynb)

📂 Project Structure
├── training_model_inception.ipynb   # Main training notebook
├── dataset/                         # Image dataset (not uploaded due to size)
├── models/                          # Saved trained models
├── README.md                        # Project documentation

🚀 How to Run

Clone the repository:

[git clone https://github.com/your-username/mango-leaf-disease-detection.git](https://github.com/Jalagamdolu/mango_leaf_disease_detection.git)
cd mango-leaf-disease-detection


Install dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook:

jupyter notebook training_model_inception.ipynb


Train the model or load a pre-trained version.

🧠 Model

Base architecture: InceptionV3

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

📊 Results

Training Accuracy: ~XX%

Validation Accuracy: ~XX%

Loss curves and accuracy plots are available in the notebook.

📸 Sample Predictions

(Add some sample images with predictions here once you have them)

🔮 Future Improvements

Deploy the model using Streamlit / Flask for real-time predictions

Extend dataset with more disease categories

Improve accuracy using transfer learning with larger models

🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit pull requests.

📜 License

This project is licensed under the MIT License.
