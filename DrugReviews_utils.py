import os
from os.path import join as oj

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    
    def __init__(self, max_features=120000, embed_size = 300, embedding_matrix = np.random.randn(120000,300)):

        super(CNN_Text, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 36
        n_classes = 14
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, n_classes)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x) 
        return logit


class BiLSTM(nn.Module):
    
    def __init__(self, max_features=120000, embed_size = 300, embedding_matrix = np.random.randn(120000,300)):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        n_classes = 14
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)


    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


MODEL_LABELS =['CNN', 'BiLSTM']
DATASIZE_LABELS = [str(0.01), str(0.1), str(1)]
from utils import cwd

def get_loaders():
   
    embed_size = 300 # how big is each word vector
    max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 750 # max number of words in a question to use
    batch_size = 512 # how many samples to process at once
    n_epochs = 5 # how many times to iterate over all samples
    n_splits = 5 # Number of K-fold Splits
    SEED = 10
    debug = 0

    data1 = pd.read_csv(oj("data","DrugReviews", "drugsComTrain_raw.csv"))

    data2 = pd.read_csv(oj("data","DrugReviews", "drugsComTest_raw.csv"))

    data = pd.concat([data1,data2])[['review','condition']]

    print(data.head())

    # remove NULL Values from data
    data = data[pd.notnull(data['review'])]


    count_df = data[['condition','review']].groupby('condition').aggregate({'review':'count'}).reset_index().sort_values('review',ascending=False)
    count_df.head()

    target_conditions = count_df[count_df['review']>3000]['condition'].values

    def condition_parser(x):
        if x in target_conditions:
            return x
        else:
            return "OTHER"
        
    data['condition'] = data['condition'].apply(lambda x: condition_parser(x))  

    data = data[data['condition']!='OTHER']


    import re

    def clean_text(x):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', x)
        return x

    def clean_numbers(x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
        return x


    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    def _get_contractions(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re
    contractions, contractions_re = _get_contractions(contraction_dict)
    def replace_contractions(text):
        def replace(match):
            return contractions[match.group(0)]
        return contractions_re.sub(replace, text)
    # Usage
    replace_contractions("this's a text with contraction")

    # lower the text
    data["review"] = data["review"].apply(lambda x: x.lower())

    # Clean the text
    data["review"] = data["review"].apply(lambda x: clean_text(x))

    # Clean numbers
    data["review"] = data["review"].apply(lambda x: clean_numbers(x))

    # Clean Contractions
    data["review"] = data["review"].apply(lambda x: replace_contractions(x))


    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(data['review'], data['condition'],
                                                        stratify=data['condition'], 
                                                        test_size=0.25)


    print("Train shape : ",train_X.shape)
    print("Test shape : ",test_X.shape)

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_y = le.fit_transform(train_y.values)
    test_y = le.transform(test_y.values)


    # ## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

    # def load_glove(word_index):
    #     EMBEDDING_FILE = 'data/glove840b300dtxt/glove.840B.300d.txt'
    #     def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    #     embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
        
    #     all_embs = np.stack(embeddings_index.values())
    #     emb_mean,emb_std = -0.005838499,0.48782197
    #     embed_size = all_embs.shape[1]

    #     nb_words = min(max_features, len(word_index)+1)
    #     embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #     for word, i in word_index.items():
    #         if i >= max_features: continue
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None: 
    #             embedding_matrix[i] = embedding_vector
    #         else:
    #             embedding_vector = embeddings_index.get(word.capitalize())
    #             if embedding_vector is not None: 
    #                 embedding_matrix[i] = embedding_vector
    #     return embedding_matrix

    # # missing entries in the embedding are set using np.random.normal so we have to seed here too

    # if debug:
    #     embedding_matrix = np.random.randn(120000,300)
    # else:
    #     embedding_matrix = load_glove(tokenizer.word_index)

    # print("Embedding shape:", np.shape(embedding_matrix))


    # Load train and test in CUDA Memory
    x_train = torch.tensor(train_X, dtype=torch.long).cuda()
    y_train = torch.tensor(train_y, dtype=torch.long).cuda()
    x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    y_cv = torch.tensor(test_y, dtype=torch.long).cuda()

    # Create Torch datasets
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    valid_set = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def get_models(individual_N=3, exp_type='models'):

    '''
    Load saved models. 

    NOTE: Change the directories to your saved models.
    
    '''

    if exp_type == 'datasets':
        models = []
        exp_dir = oj('saved_models', 'DrugReviews', 'datasets_variation', '2022-01-26-18:27')
        

        with cwd(exp_dir):
            print("Loading order of dataset proportions:", sorted(os.listdir(), key=float))                  
            for saved_dir in sorted(os.listdir(), key=float):
                for i in range(individual_N):
                    model = CNN_Text()
                    model.load_state_dict(torch.load(oj(saved_dir,'-saved_model-{}.pt'.format(i+1))))
                    models.append(model)
        return models


    elif exp_type == 'models':

        models = []
        exp_dir = oj('saved_models', 'DrugReviews', 'models_variation', '2022-01-26-13:58')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN_Text()
                model.load_state_dict(torch.load(oj('CNN_TEXT', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = BiLSTM()
                model.load_state_dict(torch.load(oj('BiLSTM', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models


    elif exp_type == 'precise':        
        models = []
        exp_dir = oj('saved_models', 'DrugReviews', 'models_variation', '2022-01-26-13:58')
        with cwd(exp_dir):
            for i in range(individual_N):
                model = CNN_Text()
                model.load_state_dict(torch.load(oj('CNN_TEXT', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = CNN_Text()
                model.load_state_dict(torch.load(oj('CNN_TEXT', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

            for i in range(individual_N):
                model = CNN_Text()
                model.load_state_dict(torch.load(oj('CNN_TEXT', '-saved_model-{}.pt'.format(i+1))))
                models.append(model)

        return models

    else:
        raise NotImplementedError(f"Experiment type: {exp_type} is not implemented.")
