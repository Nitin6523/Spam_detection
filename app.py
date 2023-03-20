import pickle
import string
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfdf= pickle.load(open('vectorizer_tf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = request.form['message']
        print(data)
        df=transform_text(data)
        print(df)
        df=[df]
        vect=tfdf.transform(df)
        print(vect)
        pred=model.predict(vect)
        print("prediction is",pred)
        return render_template('home.html',prediction=pred) 


if __name__=="__main__":
    app.run(debug=True)
