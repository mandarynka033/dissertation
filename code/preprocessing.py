# Language processing
import string
import nltk
import re
import torch

# Functions
import logging

# Windows vs Linux
import sys
path_smt = '/content/drive/MyDrive/dis/'

# Americanise
import json
with open(path_smt + 'american_british.json') as json_file:
    brit_to_amer = json.load(json_file)

# Bert
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Word Mover's Distance
from pyemd import emd
from gensim.corpora.dictionary import Dictionary
from scipy import spatial

# Preprocessing
from nltk.stem import WordNetLemmatizer
import numpy as np

# Density plot
from scipy.stats import gaussian_kde # Density plots

# Plotting
def density_plot(col, dt):
    xy = np.vstack([dt["sim"], dt[col]])
    return(gaussian_kde(xy)(xy))

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

stopwords = nltk.corpus.stopwords.words('english')
no_punct = [''.join(c for c in s if c not in string.punctuation) for s in stopwords]
add_stops = list(set(no_punct).difference(set(stopwords)))
stopwords = stopwords + add_stops
if "no" in stopwords:
    stopwords.remove("no")
if "not" in stopwords:
    stopwords.remove("not")
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def americanize(txt):
    for american_spelling, british_spelling in brit_to_amer.items():
        txt = txt.replace(british_spelling, american_spelling)
    return txt

def preprocess(col_name, datafr):
    lowered = datafr[col_name].apply(lambda x: str(x).lower())
    #     Americanize
    american = lowered.apply(americanize)
#     Removing strange apostrophes
    punct = american.str.replace(u'\u2019', " ").str.replace("’", " ").str.replace("‘", " ")
    punct2 = punct.str.replace(r'[\u201c\u201d]', '', regex = True)
    stops = punct2.apply(lambda x: ' '.join([word for word in x.replace("'", " ").split() if word not in (stopwords)]))
    #     Removing punctuation
    new_name = col_name + "_punct"
    datafr.loc[:, (new_name)] = stops.apply(remove_punctuations)

# Unnesting function
def flatten(d):
    v = [[i] if not isinstance(i, list) else flatten(i) for i in d]
    return [i for b in v for i in b]

# Removing markers from bert tokenised text
def unpad(token_text):
    unpadded = []
    for tok in token_text:
        if "[CLS]" in tok:
            tok.remove("[CLS]")
        if "[SEP]" in tok:
            tok.remove("[SEP]")
        unpadded.append(list(filter(lambda x: x != "[PAD]", tok)))
    return unpadded

# Scaling the resulting distance
def scale_func(x):
    return 1/(1+x)

def bert_preprocess(col, just_tokens = False):
    # Adding the necessary markers for Bert to work
    marked_text = "[CLS] " + col + " [SEP]"

    # Actually tokenizing text
    tokenized_text_unpad = []
    for phrase in marked_text:
        tokens = tokenizer.tokenize(phrase)
        tokenized_text_unpad.append(tokens)

    # Obtaining the length of the longest sentence
    max_len = 0
    for sent in tokenized_text_unpad:
        new_len = len(sent)
        if new_len > max_len:
            max_len = new_len

    # Padding the sentences to the maximum length
    tokenized_text = []
    for tokens in tokenized_text_unpad:
        nb_pad = max_len - len(tokens)
        padded = tokens + ["[PAD]"] * nb_pad
        tokenized_text.append(padded)

    if just_tokens:
        return tokenized_text
    else:
        # Replacing tokens with word ids
        indexed_tokens = []
        for tokens_list in tokenized_text:
            ids = tokenizer.convert_tokens_to_ids(tokens_list)
            indexed_tokens.append(ids)

        # Gathering the indices for sentences
        segments_ids = []
        for index in range(len(indexed_tokens)):
            sentence_id = [index] * len(indexed_tokens[index])
            segments_ids.append(sentence_id)

        # Changing vectors to tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensor = torch.tensor(segments_ids)

        return tokens_tensor, tokenized_text, segments_tensor

def wmdist(diction, document1, document2):
    if not document1 or not document2:
        logging.info(
            "At least one of the documents had no words that were in the vocabulary. "
            "Aborting (returning inf)."
        )
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, t1 in dictionary.items():
        if t1 not in docset1:
#             Go to next in loop
            continue

        for j, t2 in dictionary.items():
            if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                continue

            # Cosine distance makes the range of possible values bigger
            distance_matrix[i, j] = distance_matrix[j, i] = spatial.distance.cosine(diction[t1], diction[t2])

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute normalised Bag Of Words (nBOW) representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD. d1 and d2 are the positions, the distance matrix is the cost
    return emd(d1, d2, distance_matrix)
