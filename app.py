#importing all the libraries which are needed
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

#creatine a flask app named 'app'
app = Flask(__name__)

#loading the pickle file which was saved at thee training of the data it would load the CountVectorizer and our model to predict
countvector,model = pickle.load(open('model.pkl','rb'))

#route to our html file (index.html)
@app.route('/')
def home():
    return render_template('index.html')


#after the csv file is uploaded in the app on our web page and predict button is clicked below function will be executed

@app.route('/predict',methods = ['POST'])
def predict():
    #getting the files which were uploaded on our html webpage
    file = request.files['data_file']

    #reading the csv file which was uploaded and encoding it
    test= pd.read_csv(file,encoding = 'ISO-8859-1')
    
    # all the characters other than the letters are now converted into blank spaces as they are of no use
    test_transform = []
    for i in range(0,len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[i,2:27]))

    #converting the test_dataset into vectors using the countvectorizer which we saved at the time of training
    test_dataset = countvector.transform(test_transform)
    
    #predicting with the help of test dataset
    predictions = model.predict(test_dataset)
    
    #storing the prediction results 1 == up and 0 == down (market situation)
    predictions_text = []
    original_test_results=[]
    for i in range(len(predictions)):
        if predictions[i] == 1:
            predictions_text.append('up')
        else:
            predictions_text.append('down')
        
        if test['Label'][i] == 1:
            original_test_results.append('up')
        else:
            original_test_results.append('down')
        
    # list which would be exported to show on the webpage
    export_predictions=[]
    for i in range(len(predictions)):
        export_predictions.append(' the stock market as predicted will go ' + predictions_text[i] + ' and the next day it went ' + original_test_results[i])

    #calculating the f1 score
    report =  f1_score(predictions, test['Label'], average='weighted')
    
    #return the export_predictions and the f1_score
    return render_template('index.html',prediction_text = export_predictions,f1_score = report)

#main funcction to run the app
if __name__ =='__main__':
    app.run(debug = True)
