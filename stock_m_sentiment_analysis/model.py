import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#loading the dataset from github

df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Stock-Sentiment-Analysis/master/Data.csv',encoding = 'ISO-8859-1')

# dividing the data into train and test

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']



data = train.iloc[:,2:27]
data.replace('[^a-zA-Z]',' ',regex = True,inplace = True)
list1 =[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns = new_index


for index in new_index:
    data[index] = data[index].str.lower()
data.head(1)

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

print('model created0')
countvector = CountVectorizer(ngram_range = (2,2))
traindataset = countvector.fit_transform(headlines)
print('model created1')

randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
print('model created2')
with open('model.pkl', 'wb') as fout:
    pickle.dump((countvector, randomclassifier), fout)

with open('model.pkl', 'rb') as f:
    countvector, model = pickle.load(f)
print('model created3')



