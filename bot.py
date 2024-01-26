def training_model():
  import re
  import json
  import numpy as np
  import tensorflow as tf
  import nltk
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer

  intents = json.loads(open('intents.json').read())
  tags = []
  corpus = []
  y=[]

  for intent in intents['intents']:
    if intent['tag'] not in tags: tags.append(intent['tag'])

  for intent in intents['intents']:
      for pattern in intent['patterns']:
        review= re.sub('[^a-zA-Z]',' ',pattern).lower().split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
        currentIntent = [0] * len(tags)
        currentIntent[tags.index(intent['tag'])] = 1
        y.append(currentIntent)

  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X= cv.fit_transform(corpus).toarray()
  y=np.array(y)

  from sklearn.preprocessing import StandardScaler
  sc=StandardScaler()
  X=np.array(sc.fit_transform(X))

  model =  tf.keras.Sequential()
  #input Layer
  model.add(tf.keras.layers.Dense(128, input_shape=(len(X[0]),), activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  #first Hidden Layer
  model.add(tf.keras.layers.Dense(64, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  #output Layer
  model.add(tf.keras.layers.Dense(len(y[0]), activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(X,y, batch_size= 32, epochs=100)

  model.save('model.h5')
  return cv,sc,tags

def handle_message(user_message,cv,sc,tags):
  import re
  import nltk
  import json
  import random
  from tensorflow.keras.models import load_model
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer

  model = load_model('model.h5')
  intents = json.loads(open('intents.json').read())
  
  user_message = re.sub('[^a-zA-Z]', ' ', user_message).lower().split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  user_message = [ps.stem(word) for word in user_message if not word in set(all_stopwords)]
  user_message = ' '.join(user_message)
  new_corpus = [user_message]
  new_X_test = cv.transform(new_corpus).toarray()
  new_y_pred = model.predict(sc.transform(new_X_test))

  for intent in intents['intents']:
    if intent['tag'] == tags[new_y_pred[0].argmax()] :
      return (intent['responses'][random.randint(0,len(intent['responses'])-1)])