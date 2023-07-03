from tqdm import tqdm

import os, csv, json, re
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import *

tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv("data/papers_all_tags.csv")

def get_titles(df):
    keywords = [preprocess_string(k, filters=[strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]) for k in df["title"].values]
    return keywords

keywords = get_titles(df)

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
init_op = tf.global_variables_initializer()

def get_docvec_from_elmo(elmo, doc):
    embeddings = elmo(
        [doc],
        signature="default",
        as_dict=True)["elmo"]
    
    with tf.Session() as sess:
        sess.run(init_op)
        embeddings_np = sess.run(embeddings)

    docvec = embeddings_np[0].mean(axis=0)
    return docvec

docvecs = []
for i in tqdm(range(0, len(keywords), 100)):
    docs = [" ".join(k) for k in keywords[i:i+100]]
    embeddings = elmo(
        docs,
        signature="default",
        as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(init_op)
        embeddings_np = sess.run(embeddings)

    docvec = embeddings_np.mean(axis=1)
    docvecs.append(docvec)

docvecs = np.array(docvecs)
print("ELMo -> docvecs.shape:", docvecs.shape)
np.save("data/docvecs/elmo_title_cell_docvecs.npy", docvecs)

# ==============  BERT =================
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased")

docvecs = []
for keyword in tqdm(keywords):
    doc = " ".join(keyword)
    encoded_input = tokenizer(doc, return_tensors='pt')
    
    output = bert(**encoded_input)
    docvec = output[0].mean(axis=1).squeeze().detach().numpy()
    docvecs.append(docvec)

docvecs = np.array(docvecs)
print("ELMo -> docvecs.shape:", docvecs.shape)
np.save("data/docvecs/bert_avg_docvecs_cell_title.npy", docvecs)
