import flask
from flask import Flask, jsonify, request
import jsonify

import numpy as np
from string import punctuation
import pandas as pd
import re
import inflect

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('gutenberg')
nltk.download('wordnet')

import spacy
sp = spacy.load('en_core_web_sm')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

app = Flask(__name___
@app.route('/predict', methods = ['GET'])


def predict():


    class SentimentAnalysisLSTM(nn.Module):
        def __init__(self,vocab_length, embeddings_dim, n_hidden, n_layers, n_output, drop_p = 0.8):
            super().__init__()
            
            self.vocab_length = vocab_length
            self.n_layers = n_layers
            self.n_hiden = n_hidden
            self.embeddings_dim = embeddings_dim
            self.n_output = n_output
            '''
            Firstly, we define our embedding layer, which will have 
            the length of the number of words in our vocabulary
            and the size of the embedding vectors as a n_embed hyperparameter 
            to be specified  
            '''
            self.embedding = nn.Embedding(vocab_length, embeddings_dim)
            '''
            LSTM layer is defined using the output vector size from the embedding layer, 
            the length of the model's hidden state, and the number of layers that 
            our LSTM layer will have. We also add an argument to specify that our 
            LSTM can be trained on batches of data and an argument to allow us 
            to implement network regularization via dropout. 
            '''
            self.lstm = nn.LSTM(embeddings_dim, n_hidden, n_layers, batch_first = True, dropout = drop_p)
            
            self.dropout = nn.Dropout(drop_p)
            self.fc = nn.Linear(n_hidden, n_output)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, input_words):
            '''
            Next, we need to define our forward pass within our model class. 
            Within this forward pass, we just chain together the output of 
            one layer to become the input into our next layer. 
            Here, we can see that our embedding layer takes input_words as input 
            and outputs the embedded words. Then, our LSTM layer takes embedded words 
            as input and outputs lstm_out. The only nuance here is that we use view() 
            to reshape our tensors from the LSTM output to be the correct size for 
            input into our fully connected layer. The same also applies for reshaping the 
            output of our hidden layer to match that of our output node. Note that our 
            output will return a prediction for class = 0 and class = 1, 
            so we slice the output to only return a prediction for class = 1—that is, 
            the probability that our sentence is positive
            '''

            embedded_words = self.embedding(input_words)
            
            lstm_out, h = self.lstm(embedded_words)
            lstm_out = self.dropout(lstm_out)
            lstm_out = lstm_out.contiguous().view(-1, n_hidden)
            fc_out = self.fc(lstm_out)
            sigmoid_out = self.sigmoid(fc_out)
            sigmoid_out = sigmoid_out.view(batch_size, -1)
            
            sigmoid_last = sigmoid_out[:, -1]
            return sigmoid_last, h
        
        def init_hidden(self, batch_size, device = 'cpu'):
            '''
            We also define a function called init_hidden(), which initializes our hidden layer 
            with the dimensions of our batch size. This allows our model to train and predict 
            on many sentences at once, rather than just training on one sentence at a time, 
            if we so wish. Note that we define device as "cpu" here to run it on our local processor. 
            However, it is also possible to set this to a CUDA-enabled GPU in order to train it on 
            a GPU if you have one:
            '''
            self.device = device
            weights = next(self.parameters()).data
            h = (
            weights.new(n_layers, batch_size, n_hidden).zero_().to(device),
            weights.new(n_layers, batch_size, n_hidden).zero_().to(device)
                )
            return h

            
    def preprocess_review(review):
		with open('models/word_to_index.json') as handle:
			word_to_index = json.load(handle)    
        data = [[review,'X']]
        data_review = pd.DataFrame(data, columns = ['Review', 'Sentiment'])
        processed_input_review, _ = preprocessing_tokenize(data_review)
        if len(processed_input_review) > 18:
            processed_input_review = processed_input_review[:18]
        elif len(processed_input_review) < 18:
            processed_input_review = pad_text(processed_input_review,18)
        else:
            pass
        return processed_input_review
        
        
    def preprocessing_tokenize(dataframe):
        sentences = dataframe['Review']
        vocab = set()
        i = inflect.engine()
        stopwords_list = stopwords.words('english')
        processed_review = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = re.sub('[^\w\s]', '', sentence)
            tokens = word_tokenize(sentence)
            processed_sentence = []
            for token in tokens:
                if token.isalpha() and token not in stopwords_list:
                    processed_sentence.append(token)
                if token.isdigit():
                    text_digit = i.number_to_words(token)
                    text_digit = word_tokenize(text_digit)
                    processed_sentence.extend(text_digit)
            processed_review.append(processed_sentence)
            vocab.update(processed_sentence)
        return processed_review, vocab        


    def pad_text(reviews, pad_length):
        padded_reviews = []
        for review in reviews:
            l = len(review)
            if l >= pad_length:
                padded_reviews.append(review)
            else:
                pad = [''] * (pad_length - l)
                review.extend(pad)
                padded_reviews.append(review)
        return padded_reviews        

    request_json = request.get_json()
    i = request_json['input']
    
    batch_size = 1
    
    model = SentimentAnalysisLSTM(1954, 50, 100, 2, 1)
    
    model.load_state_dict(torch.load('C:\\Users\\akash\\flaskAPI\\models\S_A_LSTM.pkl')
    model.eval()
    preprocessed_review = preprocess_review(i)
    encoded_review = np.array([[ word_to_index[word] for word in sentence] for sentence in preprocessed_review])
    X_pred = torch.from_numpy(encoded_review)
    X_pred_loader = torch.utils.data.DataLoader(X_pred, batch_size = 1)
    for X in X_pred_loader:
        output = sentiment_net(X)[0].item()
	response = json.dumps({'response': output})
	return response, 200