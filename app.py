import os
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg') # Force Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask,jsonify,request,render_template,session
import joblib
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io
import base64
import seaborn as sns


app=Flask(__name__)
app.secret_key = 'your_unique_secret_key_here'

# 1.configuration
MODEL_FILE='sentiment_model.h5'
TOKENIZER_FILE='tokenizer.pkl'

# 2.load model and tokenizer
print("loading model and pipeline")
if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
    model=tf.keras.models.load_model(MODEL_FILE,compile=False)
    tokenizer=joblib.load(TOKENIZER_FILE)
    print("MODEL loaded succesfully")

else:
    print("error:model files not found .Please run main.py")
    model=None
    tokenizer=None

def improved_clean(text):
    import re
    text = str(text).lower()
    text = re.sub(r'@[^\s]+', 'user', text) 
    text = re.sub(r"http\S+", 'url', text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())    

def preprocesssing(user_text):
    # transform input data
    sequence=tokenizer.texts_to_sequences([user_text])
    padded_input=pad_sequences(sequence,maxlen=100,padding='pre')
    return padded_input 

def generate_history_scatter(history):
    
    # 1. Extract data for plotting
    plt.style.use('dark_background')
    intensities = [h['intensity'] * 100 for h in history] # Scale to 0-100
    x_axis = list(range(1, len(history) + 1))
    
    plt.figure(figsize=(7, 4), facecolor='none')
    # Use 'ggplot' style aesthetics for better visuals
    plt.plot(x_axis, intensities, marker='o', linestyle='-', color='#4caf50', linewidth=2, markersize=8)
    plt.fill_between(x_axis, intensities, color='#4caf50', alpha=0.2) # Adds "fullness"
    
    plt.title("Sentiment Intensity History", fontsize=12, pad=10)
    plt.ylim(0, 105)
    plt.xticks(x_axis) # Ensure every input has a number
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Convert to Base64
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_gradient_heatmap(model, text, tokenizer, label_index):
    words = text.lower().split()
    if not words: return ""

    # 1. Base prediction
    original_input = preprocesssing(text)
    base_probs = model.predict(original_input)[0][0]
    base_score = base_probs[label_index]

    word_data = []
    
    # 2. Leave-One-Out Sensitivity Analysis
    for i in range(len(words)):
        temp_words = words[:i] + words[i+1:]
        temp_text = " ".join(temp_words)
        temp_input = preprocesssing(temp_text)
        new_probs = model.predict(temp_input)[0][0]
        
        # Calculate impact on the specific predicted class
        impact = base_score - new_probs[label_index]
        word_data.append({'word': words[i], 'impact': impact})

    # 3. Filter Top 7 and Calculate Percentages
    # Sort by absolute impact to find the most influential words
    word_data.sort(key=lambda x: abs(x['impact']), reverse=True)
    top_7 = word_data[:7]
    
    total_impact = sum(abs(item['impact']) for item in top_7)
    
    display_labels = []
    display_scores = []
    
    for item in top_7:
        # Calculate percentage contribution
        percent = (abs(item['impact']) / total_impact * 100) if total_impact > 0 else 0
        display_labels.append(f"{item['word']}\n({percent:.1f}%)")
        
        # Color Logic: Positive impact = Green (positive value), Negative = Red (negative value)
        # We use a scale of -1 to 1 for the heatmap color mapping
        display_scores.append(item['impact'])

    # 4. Generate Color-Coded Heatmap
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 3), facecolor='none')
    
    # Use 'RdYlGn' (Red-Yellow-Green) diverging colormap
    sns.heatmap(np.array(display_scores).reshape(1, -1), 
                annot=np.array(display_labels).reshape(1, -1), 
                fmt="", cmap="RdYlGn", center=0, cbar=False, 
                xticklabels=False, yticklabels=False,
                annot_kws={"size": 12, "weight": "bold", "color": "black"})
    
    plt.title("Top Word Contributions (%)", pad=15, fontsize=14)
    plt.tight_layout()
    
    # 5. Base64 Conversion
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/',methods=['GET'])
def show():
    return render_template('index.html')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({'error':'model is not loaded'}),500
    
    try:
        data=request.form['text']
        user_text=str(data)
        
        clean_text=improved_clean(user_text)
        # 1.preprocess
        tokenization_text=preprocesssing(clean_text)

        # 2 predict
        predictions=model.predict(tokenization_text)
        # Process Sentiment
        sentiment_labels = ['Negative', 'Neutral', 'Positive']      #
        predicted_class =predictions[0][0]                              #
        predicted_idx=np.argmax(predicted_class) 
        sentiment=sentiment_labels[predicted_idx]
        intensity=float(predictions[1][0][0])
        #3 confidence percentage
        confidence=float(predicted_class[predicted_idx] *100) 

        
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({'intensity': float(intensity), 'label': sentiment})
        if len(session['history']) > 10:
            session['history'].pop(0)

        # 3. Generate Visuals
        # Only generate heatmap if user wants highlights or for every predict
        heatmap_url = generate_gradient_heatmap(model, user_text, tokenizer, predicted_idx)
        history_plot = generate_history_scatter(session['history'])

        # 4. Return to Template
        return render_template('index.html', 
                               sentiment=sentiment, 
                               intensity=round(intensity,2), 
                               text=user_text, 
                               confidence=round(confidence,2),
                               heatmap_url=heatmap_url,
                               history_plot=history_plot)


    except Exception as e:
        return jsonify({'error':str(e)}),400

    
if __name__=='__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",7860)))


