from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
app = Flask(__name__)
countvector,model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    file = request.files['data_file']
    #print(test)
    #print(file_contents)
    test= pd.read_csv(file,encoding = 'ISO-8859-1')
     
    print(test)
    print('done')
    
    test_transform = []

    for i in range(0,len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[i,2:27]))


    test_dataset = countvector.transform(test_transform)
    
    
    predictions = model.predict(test_dataset)
    #print(predictions)
    predictions_text = []
    ans = []
    for i in range(len(predictions)):
        if predictions[i] == 1:
            ans.append('up')
        else:
            ans.append('down')
    ans1=[]
    for i in range(len(test['Label'])):
        if predictions[i] == 1:
            ans1.append('up')
        else:
            ans1.append('down')
    
    for i in range(len(predictions)):
        predictions_text.append(' the stock market as predicted will go ' + ans[i] + ' and the next day it went ' + ans1[i])

    report =  f1_score(predictions, test['Label'], average='weighted')
    print(report)
    return render_template('index.html',prediction_text = predictions_text,f1_score = report)


if __name__ =='__main__':
    app.run(debug = True)
