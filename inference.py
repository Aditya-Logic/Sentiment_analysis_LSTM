import os
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask,jsonify,request,render_template
import joblib
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app=Flask(__name__)

# 1.configuration
MODEL_FILE='sentiment_model.keras'
TOKENIZER_FILE='tokenizer.pkl'

# 2.load model and tokenizer
print("loading model and pipeline")
if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
    model=tf.keras.models.load_model(MODEL_FILE)
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
        predicted_class = np.argmax(predictions[0][0])                                #       

        return render_template('index.html',sentiment=sentiment_labels[predicted_class],intensity=round(predictions[1][0][0],2),text=user_text)


    except Exception as e:
        return jsonify({'error':str(e)}),400

    
if __name__=='__main__':
    app.run(debug=True)


