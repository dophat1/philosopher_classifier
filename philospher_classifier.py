import string 
import numpy as np
# Tokenizing and word processing

def tokenize(text):
    clean_text = text.lower()
    clean_text = ''.join([character for character in clean_text if character not in string.punctuation])
    return clean_text.split()

def build_vocab(texts):

    all_word = []

    for text in texts:
        all_word += tokenize(text)
    
    vocabulary = set(all_word)
    dictionary = {word: index for index, word in enumerate(vocabulary)}
    
    return dictionary
    
def vectorize(text, vocabulary):
    clean_text = tokenize(text)
    vector = [0 for i in range(len(vocabulary))]
    for word in clean_text:
        if word in vocabulary:
            vector[vocabulary[word]] += 1 

    return vector
 

# Building neural network

class MLP: 
    def __init__(self, input_size, hidden_size, output_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.random.randn(self.hidden_size,)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.random.randn(self.output_size,)

    def forward(self, x):
        self.h = relu(np.dot(x, self.W1) + self.b1)
        self.y = softmax(np.dot(self.h,self.W2) + self.b2)
        return self.y


# ReLu 
def relu(x):
    return np.maximum(x, 0)
     
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def loss(y_pred, y_true):
    loss = -np.log(y_pred[y_true])
    return loss
     