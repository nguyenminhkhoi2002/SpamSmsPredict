import pandas as pd
import nltk
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import re
import pickle

#Get Sms Dataset
sms = pd.read_csv('E:/sms-spam-python/smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])
sms.drop_duplicates(inplace=True)
sms.reset_index(drop=True, inplace=True)

#Cleaning the messages
corpus = []
ps = PorterStemmer()
for i in range(0,sms.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.message[i]) #Cleaning special character from the message
    message = message.lower() #Converting the entire message into lower case
    words = message.split() # Tokenizing the review by words
    words = [word for word in words if word not in set(stopwords.words('english'))] #Removing the stop words
    words = [ps.stem(word) for word in words] #Stemming the words
    message = ' '.join(words) #Joining the stemmed words
    corpus.append(message) #Building a corpus of messages

#Save corpus for use in deployment
file_name = "corpus3.pkl"
pickle.dump(corpus, open(file_name, 'wb'))

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

#Extracting dependent variable from the dataset
y = pd.get_dummies(sms['label'])
y = y.iloc[:, 1].values

#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
 
tuned_parameters = {'kernel':['linear', 'rbf'], 'gamma':[1e-3, 1e-4], 'C':[1,10,100,1000]}
 
classifier = GridSearchCV(SVC(), tuned_parameters)
classifier.fit(X_train, y_train)

#Save Model
file_name = "svm.pkl"
pickle.dump(classifier, open(file_name, 'wb'))