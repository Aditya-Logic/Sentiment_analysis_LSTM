# 🧠 Sentiment Analysis Web App (Multi-Task Deep Learning with Flask)

## 📌 Project Overview

This project is an **Advanced Sentiment Analysis Web Application** that uses a **Multi-Task Deep Learning model** built with **TensorFlow/Keras**.

The system predicts:

1. **Sentiment Category** → Negative / Neutral / Positive
2. **Sentiment Intensity** → Strength of emotion (0.0 – 1.0)

The model uses:

* Bidirectional LSTM neural networks
* Pretrained **GloVe word embeddings**
* Multi-head architecture (classification + regression)
* Flask web interface for real-time predictions

This project demonstrates an **end-to-end NLP pipeline** from training to deployment.

---
## 🚀 Live Demo

You can try the application here:

👉 **Live App:**  
https://huggingface.co/spaces/Andy12vb/sentiment-analysis-lstm

## 🚀 Features

✅ Multi-task learning (sentiment + intensity)
✅ Pretrained GloVe embeddings integration
✅ Bidirectional LSTM architecture
✅ Early stopping & learning rate scheduling
✅ Text preprocessing & tokenization
✅ Flask web interface
✅ Model persistence (.keras + tokenizer.pkl)

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* VADER Sentiment (for intensity labels)
* Flask
* Joblib

---

## 📂 Project Structure

```
project/
│── main_new.py                # Model training script
│── app.py               # Flask web application
│── sentiment_model.h5      # Saved trained model
│── tokenizer.pkl              # Saved tokenizer
│── cleaned_twitter_data.csv   # Dataset
│── glove.6B.100d.txt          # Pretrained embeddings
│── templates/
│     ├── index.html
│     └── about.html
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Aditya-Logic/Sentiment_analysis_LSTM.git
cd Sentiment_anlaysis_LSTM
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is not available:

```bash
pip install tensorflow flask pandas numpy scikit-learn joblib vaderSentiment
```

---

## 📊 Model Architecture

### Input

* Tokenized text sequences (max length = 100)

### Embedding Layer

* Pretrained **GloVe 100-dimensional embeddings**
* Trainable for better generalization

### Shared Layers

* Spatial Dropout
* Bidirectional LSTM (64 units)
* Bidirectional LSTM (32 units)

### Output Heads

1. **Sentiment Head**

   * Dense + Dropout
   * Softmax (3 classes)

2. **Intensity Head**

   * Dense layers
   * Sigmoid output (0–1 range)

---

## 🏋️ Model Training

Run:

```bash
python main_new.py
```

This will:

* Train the neural network
* Save model → `sentiment_model.h5`
* Save tokenizer → `tokenizer.pkl`

---

## 🌐 Running the Web Application

Start Flask server:

```bash
python app.py
```

Then open browser:

```
http://127.0.0.1:5000
```

---

## 📝 Prediction Example

Input:

```
I absolutely love this product, it is amazing!
```

Output:

```
Sentiment: Positive
Intensity: 0.92
```

---

## 🔍 API Workflow

1. User enters text
2. Text cleaning & preprocessing
3. Tokenization + padding
4. Model prediction
5. Sentiment + intensity displayed on webpage

---

## 🔮 Future Improvements

* Deploy on Render / AWS / Docker
* Add attention mechanism
* Real-time REST API endpoint
* Emotion classification (anger, joy, sadness)
* Transformer-based model (BERT)

---

## 👨‍💻 Author

Aditya Verma

---

## 📜 License

This project is open source and available under the MIT License.

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
