# importing the pandas library
import pandas as pd

# Reading the dataset
df = pd.read_csv("Tweets.csv")

# Only text, airline_sentiment columns
review_df = df[['text','airline_sentiment']]
#review_df = review_df.head(6600)
#print(review_df)

# Remove neutral airline_sentiment rows
review_df =review_df[review_df['airline_sentiment'] != 'neutral']

# Checking values of airline_sentiment
print(review_df["airline_sentiment"].value_counts())

# Convert the categorical values to numeric using the factorize() method.
sentiment_label = review_df.airline_sentiment.factorize()
print(sentiment_label)

#     Convert the text into an array of vector embeddings
# Representing the relationship between the words in the text
# first give each of the unique words a unique number and
# then replace that word with the number assigned

# First, retrieve all the text data from the dataset
tweet = review_df.text.values

# then tokenize all the words
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
# The fit_on_texts() method creates an association between the words and the assigned numbers
tokenizer.fit_on_texts(tweet)

# replace the words with their assigned numbers using the text_to_sequence() method
encoded_docs = tokenizer.texts_to_sequences(tweet)

#  to pad the sentences to have equal length
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

#      Build the Text Classifier

#   Creates a robust/strong model avoiding overfitting
# Dropout is one of the regularization techniques
# we drop some neurons randomly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(len(padded_sequence), embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())

#    Train the sentiment analysis model
# Train the sentiment analysis model for
# 5 epochs on the whole dataset
#  with a batch size of 32
#  and a validation split of 20%
history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

#  plot these obtained metrics using the matplotlib
import matplotlib.pyplot as plt
#accuracy
plt.plot(history.history['accuracy'], label='acc') #training set accuracy
plt.plot(history.history['val_accuracy'], label='val_acc') #test set accuracy
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")
#loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plt.jpg")

#     Execute sentiment analysis model
# prediction function
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])

#executing prediction function
#q              test_sentence1 = "I enjoyed my journey on this flight."
val = ""
while val != "0":
    val = input("enter your value")
    predict_sentiment(val)

#test_sentence2 = "This is the worst flight experience of my life!"
#predict_sentiment(test_sentence2)

#################################################################################################################









# reading the csv file
#df = pd.read_csv("data.csv")

# updating the column value/data
#df['diagnosis'] = df['diagnosis'].replace({'B': 1})

# writing into the file
#df.to_csv("data.csv", index=False)

#print(df)

# Drop last column of a dataframe
#df.iloc[row_start:row_end , col_start, col_end]
#df = df.iloc[: , :-1]