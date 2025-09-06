import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
import random

import matplotlib.pyplot as plt

# Rest of the code remains the same until loading the training data...

# ... (previous code)
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
with open('intents.json','r') as data_file:
    data_file = json.load(data_file)

for intent in data_file['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
# ... (previous code)

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# Add a dictionary to keep track of patterns with their corresponding tags
pattern_tags = {}

for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Check if the pattern has a context
    if 'context' in data_file['intents'][classes.index(doc[1])]:
        context = data_file['intents'][classes.index(doc[1])]['context']
    else:
        context = None

    training.append([bag, output_row, context])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# Separate the training data into input features (X) and output labels (Y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

print("Training data created")

# Create and compile the Bidirectional LSTM model
model = Sequential()
model.add(Embedding(len(words), 128, input_length=len(train_x[0])))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model using Adam optimizer and categorical cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Fit the model to the training data
hist = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_bilstm.h5', hist)

print("Bidirectional LSTM model created")

# Print average accuracy
avg_accuracy = np.mean(hist.history['accuracy'])
print("Average training accuracy:", avg_accuracy)


# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Bidirectional LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Bidirectional LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved LSTM model
from keras.models import load_model

lstm_model = load_model('chatbot_model_lstm.h5')

# Make predictions on the validation data
val_lstm_predictions = lstm_model.predict(np.array(val_x))

# Convert one-hot encoded predictions to class labels
val_y_labels = np.argmax(val_y, axis=1)
val_lstm_pred_labels = np.argmax(val_lstm_predictions, axis=1)

# Compute and visualize the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix_lstm = confusion_matrix(val_y_labels, val_lstm_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lstm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (LSTM)')
plt.show()



