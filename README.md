# Emotion Detection Text Classification

This project focuses on classifying textual data into six distinct emotions using transformer-based models. It includes data preprocessing, model training, evaluation, and a user-friendly Streamlit application for real-time emotion detection.

---

## 📁 Project Directory Structure

```
Emotion_Detection_Text_Classification/
├── emotion_model/              # Saved transformer model and tokenizer
├── plots/                      # Visualizations (e.g., confusion matrix)
├── app.py                      # Streamlit application for emotion detection
├── EDA.ipynb                   # Exploratory Data Analysis notebook
├── train_model.ipynb           # Model training and evaluation notebook
├── README.md                   # Read me file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── .gitattributes              # Git attributes file
```

---

## 🚀 Features

* **Transformer-Based Model**: Utilizes Hugging Face's `transformers` library for state-of-the-art text classification.
* **Efficient Data Handling**: Employs `DataCollatorWithPadding` for dynamic padding, optimizing memory usage during training.
* **GPU Acceleration**: Training on GPU significantly reduces time (\~6 minutes). Note: CPU training may take several hours.
* **Interactive Web App**: A Streamlit application (`app.py`) allows users to input text and receive emotion predictions in real-time.

---

## 📊 Dataset Overview

The [dataset](https://huggingface.co/Worldman/distilbert-base-uncased-finetuned-emotion/commit/c1eced6d784d26af260d9c519f2cd4aae46b4602) comprises labeled textual data categorized into six emotions:

- **0** : "sadness"
- **1** : "joy"
- **2** : "love"
-  **3** : "anger"
- **4** : "fear"
- **5** : "surprise"

Each entry in the dataset includes a text snippet and its corresponding emotion label.

---

## 🧠 Model Architecture

* **Base Model**: Transformer-based architecture from Hugging Face's `transformers` library.
* **Tokenizer**: Corresponding tokenizer for the selected transformer model.
* **Training**: Implemented in `train_model.ipynb`, leveraging GPU for accelerated training.
* **Evaluation**: Performance metrics and confusion matrix are generated post-training.

The model achieved an accuracy of ``0.94`` on test data.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Saber0722/Emotion_Detection_Text_Classification.git
cd Emotion_Detection_Text_Classification
```

### 2. (Optional) Create a Virtual Environment

While optional, it's recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

Execute the training notebook:

```bash
jupyter notebook train_model.ipynb
```

*Note*: Training on GPU is highly recommended for efficiency.

### 5. Run the Streamlit App

Launch the web application:

*Note*: Running the application for the first time will take some time and this time will signficantly reduce from the next time as the cached data will be used.

```bash
streamlit run app.py
```

Access the app in your browser at `http://localhost:8501/`.

---

## 📈 Exploratory Data Analysis

The `EDA.ipynb` notebook provides insights into the dataset, including:

* Distribution of emotion labels
* Text length analysis

---

## 📷 Visualizations

The `plots/` directory contains visual representations of model performance, including the confusion matrix (`confusion_matrix.png`).

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
