import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping   
from sklearn.model_selection import train_test_split
import pickle

sms = pd.read_csv('E:/sms-spam-python/smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])
sms.drop_duplicates(inplace=True)
sms.reset_index(drop=True, inplace=True)

sms['label'] = sms.label.map({'ham':0, 'spam':1})

#Split data into training and test sets
X = sms['message'].values
y = sms['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Prepare the tokenizer
t = Tokenizer()
t.fit_on_texts(X_train)

#Integer encode the documents
encoded_train = t.texts_to_sequences(X_train)
encoded_test = t.texts_to_sequences(X_test)

#Pad documents to a max length = 8 words
max_length = 8
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

#Calculate the vocabulary size
vocab_size = len(t.word_index) + 1

#Define the model
model = Sequential()
model.add(Embedding(vocab_size, 24, input_length=max_length))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compile the model
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

#Define early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

#Fit the model
model.fit(x=padded_train, y=y_train, epochs=50, validation_data=(padded_test, y_test), verbose=1, callbacks=[early_stop])

#Save Model
model.save('backpropagation')
with open("backpropagation/tokenizer.pkl", "wb") as output:
    pickle.dump(t, output, pickle.HIGHEST_PROTOCOL)