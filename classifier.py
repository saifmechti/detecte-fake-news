### import libraries
import pandas as pd
import numpy as np


## import train/test data
df = pd.read_excel('data/data.xlsx', sheet_name='Labelled_Data')
df = df[['Tweet Text','Is_Bot ( 1 for Bot / 0 for Human)']]
df = df.rename(columns={'Tweet Text': "tweet", 'Is_Bot ( 1 for Bot / 0 for Human)': "target"})

df2 = pd.read_excel('data/data.xlsx', sheet_name='Test_Data')
df2 = df2[['Tweet Text','Is_Bot ( 1 for Bot / 0 for Human)']]
df2 = df2.rename(columns={'Tweet Text': "tweet", 'Is_Bot ( 1 for Bot / 0 for Human)': "target"})


##### train LSTM

sentences_train = df.tweet
sentences_test = df2.tweet
y_train = df.target
y_test = df2.target

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index) + 1

from keras.preprocessing.sequence import pad_sequences

#max_len = max([len(i) for i in X_train])
#print(max_len)

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


from keras.models import Sequential
from keras import layers
from keras.layers import Dense, LSTM

embedding_dim = 1000

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Calculate the weights for each class so that we can balance the data
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

## Fit the model
#model.fit(X_train, y_train, validation_split=0.3, epochs=5) #, class_weight=weights
model.fit(X_train, y_train,epochs=5, validation_data=(X_test, y_test), batch_size=128)
###Test model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))



from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
matrix = confusion_matrix(y_test, y_pred)
