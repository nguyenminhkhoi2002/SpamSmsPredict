from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pickle 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Import Pickle file
file_name1 = "bayes.pkl"
classifier1 = pickle.load(open(file_name1, 'rb'))
file_name1 = "corpus1.pkl"
corpus1 = pickle.load(open(file_name1, 'rb'))

with open("backpropagation/tokenizer.pkl", "rb") as input:
    tokenizer = pickle.load(input)

file_name3 = "svm.pkl"
classifier3 = pickle.load(open(file_name3, 'rb'))
file_name3 = "corpus3.pkl"
corpus3 = pickle.load(open(file_name3, 'rb'))


#Load Bayes model
cv1 = CountVectorizer(max_features=2500)
X = cv1.fit_transform(corpus1).toarray()

def predict_spam_bayes(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv1.transform([final_message]).toarray()
    return classifier1.predict(temp)

#Load Backpropagation model
def predict_spam_backpropagation(sample_message):
    max_length = 8
    message = tokenizer.texts_to_sequences([sample_message])
    message = pad_sequences(message, maxlen=max_length, padding='post')
    model = tf.keras.models.load_model('backpropagation')
    result = (model.predict(message) > 0.5).astype("int32").item()
    print(result)
    return result

#Load SVM model
cv3 = CountVectorizer(max_features=2500)
X = cv3.fit_transform(corpus3).toarray()

def predict_spam_svm(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv3.transform([final_message]).toarray()
    return classifier3.predict(temp)

# Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:bmqisme123@localhost:3306/sms_spam'
db = SQLAlchemy(app)

#Database
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255))
    available = db.Column(db.Integer)
    bayesResult = db.Column(db.Integer)
    backpropagationResult = db.Column(db.Integer)
    svmResult = db.Column(db.Integer)


#Add message to database and send result back
@app.route('/api/v1', methods=['POST'])
def add_message():
    data = request.json
    if data['message'] == "":
        return jsonify({'message': 'message is empty'}), 400
    
    print(data['message'])

    checkResultBayes = predict_spam_bayes(data['message'])
    if checkResultBayes == 1:
        checkResultBayes = 1
    else:
        checkResultBayes = 0

    checkResultBackpropagation = predict_spam_backpropagation(data['message'])
    if checkResultBackpropagation == 1:
        checkResultBackpropagation = 1
    else:
        checkResultBackpropagation = 0

    checkResultSVM = predict_spam_svm(data['message'])
    if checkResultSVM == 1:
        checkResultSVM = 1
    else:
        checkResultSVM = 0

    message = Message(message=data['message'], available=data['available'], bayesResult=checkResultBayes, backpropagationResult=checkResultBackpropagation, svmResult=checkResultSVM)
    db.session.add(message)
    db.session.commit()
    print(message)
    return jsonify({'bayesResult': message.bayesResult, 'backpropagationResult': message.backpropagationResult, "svmResult": message.svmResult}), 200

#Get all messages
@app.route('/api/v1', methods=['GET'])
def get_messages():
    messages = Message.query.all()
    result = []
    for message in messages:
        if message.available == 1:
            result.append({'id': message.id, 'message': message.message, 'available': message.available, 'bayesResult': message.bayesResult, 'backpropagationResult': message.backpropagationResult, "svmResult": message.svmResult})
    return jsonify(result), 200  

#Set unavailable
@app.route('/api/v1/<int:id>', methods=['POST'])
def set_available(id):
    message = Message.query.get(id)
    message.available = 0
    db.session.commit()
    return jsonify({'id': message.id, 'message': message.message, 'available': message.available, 'bayesResult': message.bayesResult, 'backpropagationResult': message.backpropagationResult, "svmResult": message.svmResult}), 200

#Set all unavailable
@app.route('/api/v1/clear', methods=['POST'])
def set_all_unavailable():
    messages = Message.query.all()
    for message in messages:
        message.available = 0
    db.session.commit()
    return jsonify({'message': 'All messages are not available'}), 200

if __name__ == '__main__':
    app.run(debug=True)