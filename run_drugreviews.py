
import time
import pandas as pd
import numpy as np
import re

#import spacy
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

import os 
from os.path import join as oj

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from multiprocessing import  Pool
import numpy as np


import time
import datetime

from DrugReviews_utils import CNN_Text, BiLSTM


from utils_ml import train, test

def train_store_models(train_loader, test_loader, num_models=5, method='CNN_TEXT', n_workers=4, epoch=30):

    if method == 'CNN_TEXT':
        original_model = CNN_Text(embedding_matrix=embedding_matrix)
    else:
        original_model = BiLSTM(embedding_matrix=embedding_matrix)

    ML_models = []
    optimizers = []

    for _ in range(num_models):

        model = deepcopy(original_model).to(torch.device('cuda'))
        ML_models.append(model)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        optimizers.append(optimizer)


    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer in zip(ML_models, optimizers) ]
        output = pool.starmap(train, input_arguments)

    ML_models = output

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        output = pool.starmap(test, input_arguments)
    print("Test accuracies for {}:".format(method), output)

    os.makedirs(method, exist_ok=True)

    for i,model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(method, '-saved_model-{}.pt'.format(i+1)))

    return


def train_store_models_datasets(train_loader, test_loader, dataset_proportion=1, num_models=5, method='CNN_TEXT', n_workers=4, epoch=50):
    
    if method == 'CNN_TEXT':
        original_model = CNN_Text(embedding_matrix=embedding_matrix)
    else:
        original_model = BiLSTM(embedding_matrix=embedding_matrix)

    ML_models = []
    optimizers = []

    for _ in range(num_models):

        # if torch.cuda.device_count() > 1:
            # model = deepcopy(nn.DataParallel(original_model)).to(torch.device('cuda'))
        # else:
        model = deepcopy(original_model).to(torch.device('cuda'))


        ML_models.append(model)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        optimizers.append(optimizer)
    
    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(train_loader), optimizer, epoch) for model, optimizer in zip(ML_models, optimizers) ]
        output = pool.starmap(train, input_arguments)

    ML_models = output

    with Pool(n_workers) as pool:
        input_arguments = [(model, torch.device('cuda'), deepcopy(test_loader)) for model in ML_models ]
        output = pool.starmap(test, input_arguments)
    print("Test accuracies for {}:".format(method), output)


    os.makedirs(str(dataset_proportion), exist_ok=True)

    for i,model in enumerate(ML_models):
        torch.save(model.state_dict(), oj(str(dataset_proportion), '-saved_model-{}.pt'.format(i+1)))

    return


## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule
def load_glove(word_index):
    EMBEDDING_FILE = 'data/glove840b300dtxt/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

import os
from os.path import join as oj
import time
import datetime
import argparse
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import numpy as np

from utils import cwd

if __name__ == '__main__':

    embed_size = 300 # how big is each word vector
    max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 750 # max number of words in a question to use
    batch_size = 512 # how many samples to process at once
    n_epochs = 5 # how many times to iterate over all samples
    n_splits = 5 # Number of K-fold Splits
    SEED = 10
    debug = 0
    np.random.seed(SEED)


    data1 = pd.read_csv(oj("data","DrugReviews", "drugsComTrain_raw.csv"))

    data2 = pd.read_csv(oj("data","DrugReviews", "drugsComTest_raw.csv"))

    data = pd.concat([data1,data2])[['review','condition']]


    # remove NULL Values from data
    data = data[pd.notnull(data['review'])]

    count_df = data[['condition','review']].groupby('condition').aggregate({'review':'count'}).reset_index().sort_values('review',ascending=False)

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


    # missing entries in the embedding are set using np.random.normal so we have to seed here too
    if debug:
        embedding_matrix = np.random.randn(120000,300)
    else:
        embedding_matrix = load_glove(tokenizer.word_index)
    print("Embedding shape:" , embedding_matrix.shape)

    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)
    x_cv = torch.tensor(test_X, dtype=torch.long)
    y_cv = torch.tensor(test_y, dtype=torch.long)


    # Create Torch datasets
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    valid_set = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)


    parser = argparse.ArgumentParser(description='Process which type of training to conduct.')
    parser.add_argument('-N', '--num_models', help='The number of models for a class of model or a type of training.', type=int, default=3)
    parser.add_argument('-t', '--type', help='The type of experiments.', type=str, default='models', choices=['datasets', 'models'])


    args = parser.parse_args()
    print(args)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')


    if args.type == 'models':

        exp_dir = oj('saved_models', 'DrugReviews', 'models_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='CNN_TEXT' )
            train_store_models(train_loader, test_loader, num_models=args.num_models, method='BiLSTM', n_workers=1)

    elif args.type == 'datasets':

        exp_dir = oj('saved_models', 'DrugReviews', 'datasets_variation', st)
        os.makedirs(exp_dir, exist_ok=True)
        with cwd(exp_dir):

            dataset_proportion = 0.01
            smallest = list(range(0, len(train_set), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_set, smallest)
            train_loader_smallest = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smallest), len(train_loader_smallest), dataset_proportion))
            train_store_models_datasets(train_loader_smallest, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN_TEXT', epoch=30)

            dataset_proportion = 0.1
            smaller = list(range(0, len(train_set), int(1//dataset_proportion)))
            trainset_1 = torch.utils.data.Subset(train_set, smaller)
            train_loader_smaller = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, num_workers=1, pin_memory=True)
            print('Length of dataset {}, loader {}, for proporation {}'.format(len(smaller), len(train_loader_smaller), dataset_proportion))
            train_store_models_datasets(train_loader_smaller, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN_TEXT', epoch=30)

            dataset_proportion = 1
            print('Length of dataset {} for proporation {}'.format(len(train_set), dataset_proportion))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, pin_memory=True)
            train_store_models_datasets(train_loader, test_loader, dataset_proportion=dataset_proportion, num_models=args.num_models, method='CNN_TEXT' , epoch=30)
