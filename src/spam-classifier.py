# Importing dataset
import pandas as pd

msg = pd.read_csv('../data/SMSSpamCollection', sep='\t', names=['labels', 'messages'])

# Data cleaning and Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wnl = WordNetLemmatizer()

sw = stopwords.words('english')

stem_corpus = []
lemmatize_corpus = []
for i in range(len(msg)):
    res = re.sub('[^a-zA-z]', ' ', msg['messages'][i])
    res = res.lower().split()
    
    # stemming
    t1 = [ps.stem(word) for word in res if word not in sw]
    t1 = ' '.join(t1)
    stem_corpus.append(t1)
    
    # lemmatizing
    t2 = [wnl.lemmatize(word) for word in res if word not in sw]
    t2 = ' '.join(t2)
    lemmatize_corpus.append(t2)

# Creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

X1 = cv.fit_transform(stem_corpus).toarray()
X2 = cv.fit_transform(lemmatize_corpus).toarray()

y1 = pd.get_dummies(msg['labels'])
y1 = y1.iloc[:,1].values

y2 = pd.get_dummies(msg['labels'])
y2 = y2.iloc[:,1].values

# Test and Train split
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.20, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.20, random_state=0)

# Model Training - using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

stem_model = MultinomialNB().fit(X_train1, y_train1)
lemm_model = MultinomialNB().fit(X_train2, y_train2)

y_pred1 = stem_model.predict(X_test1)
y_pred2 = lemm_model.predict(X_test2)

# Metrics and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

conf_mat1 = confusion_matrix(y_test1, y_pred1)
conf_mat2 = confusion_matrix(y_test2, y_pred2)

stem_acc = accuracy_score(y_test1, y_pred1)
lemm_acc = accuracy_score(y_test2, y_pred2)

# Results
print(f'Accuracy using stemming is {round(stem_acc*100, 3)}%')
print(f'Accuracy using lemmatization is {round(lemm_acc*100, 3)}%')