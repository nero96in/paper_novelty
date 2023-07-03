# from .modules import common

import os, json, datetime, re, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from collections import Counter
from numpy import dot
from numpy.linalg import norm
import numpy as np

from gensim.models import fasttext
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
from gensim import matutils
from gensim.parsing.preprocessing import *

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from scipy.stats import mannwhitneyu

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year_start", dest="year_start", action="store")
parser.add_argument("--year_end", dest="year_end", action="store")
args = parser.parse_args()

year_start = int(args.year_start)
year_end = int(args.year_end)

DATA_DIR = "data"
RESULTS_DIR = "results"
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
PAPERS_DETAILS_DIR = os.path.join(PAPERS_DIR, "details")

PAPERS_ALL_PATH = os.path.join(PAPERS_DIR, "cell_all.csv")
PAPERS_RECOM_PATH = os.path.join(PAPERS_DIR, "cell_recom.csv")

papers_all = pd.read_csv(PAPERS_ALL_PATH)
papers_all = papers_all[papers_all['type'] == 'Article']
papers_all["PUBMED ID"] = papers_all["PUBMED ID"].fillna(-1).astype(int)

papers_recom = pd.read_csv(PAPERS_RECOM_PATH)
papers_recom = papers_recom.rename(columns={"pubMedId": "PUBMED ID"})

df_tags = []
for id in tqdm(papers_recom.id):
    tags = {
        'id': id,
        'CONTROVERSIAL': 0,
        'GOOD_FOR_TEACHING': 0,
        'CONFIRMATION': 0,
        'TECHNICAL_ADVANCE': 0,
        'NOVEL_DRUG_TARGET': 0,
        'NEW_FINDING': 0,
        'HYPOTHESIS': 0,
        'REFUTATION': 0,
        'NEGATIVE': 0,
    }
    with open(os.path.join(PAPERS_DETAILS_DIR, f"{id}.json"), "r") as f: data = json.load(f)
    for evaluation in data['pageProps']['initialArticle']['evaluations']:
        for tag in evaluation['classifications']:
            tags[tag['code']] += 1
            
    df_tags.append(tags)
    
df_tags = pd.DataFrame(df_tags)
papers_recom = pd.merge(papers_recom, df_tags, how="inner", on="id")

papers_all_tags = pd.merge(papers_all, papers_recom.drop(columns=["title", "authors", "openAccess", "publishedDateForPrint"]), how="left", on="PUBMED ID")
papers_all_tags[papers_all_tags.columns[-9:]] = papers_all_tags[papers_all_tags.columns[-9:]].fillna(0)
papers_all_tags["date"] = pd.to_datetime(papers_all_tags["date"])
papers_all_tags["year"] = papers_all_tags["date"].dt.year
papers_all_tags = papers_all_tags[papers_all_tags.year < 2022]

def keywords2avg_distance(model, keywords):
    '''
        키워드 벡터들을 입력했을 때, 이의 평균 벡터를 반환하여 한 문서의 대표 벡터를 추출하는 함수

        - model: 사용할 모델
        - keywords: 하나의 문서에 포함된 키워드들
    '''
    def cos_sim(v1, v2): return dot(v1, v2)/(norm(v1)*norm(v2))

    results = []
    for doc in tqdm(keywords):
        if len(doc) < 2: continue
        vectors = []
        for k in doc: vectors.append(model.wv[k])

        total = []
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                if i >= j: continue
                distance = cos_sim(v1, v2)
                total.append(distance)

        avg_distance = sum(total) / len(total)
        results.append(avg_distance)
    results = sum(results) / len(results)
    print(results)

    return results

def keywords2vec(model, keywords):
    '''
        키워드 벡터들을 입력했을 때, 이의 평균 벡터를 반환하여 한 문서의 대표 벡터를 추출하는 함수
        
        - model: 사용할 모델
        - keywords: 하나의 문서에 포함된 키워드들
    '''
    vectors = []
    for k in keywords:
        vectors.append(model.wv[k])
    vectors = np.array(vectors)
    
    repr_vec = vectors.mean(axis=0)
    return repr_vec

def get_keywords(df, year, until=False, column="keywords"):
    '''
        게재 일자가 포함된 서지 정보 데이터에서 특정 연도 이전(혹은 당해)의 문서 인덱스와 키워드 셋을 반환하는 함수.
        
        - df: 서지정보 데이터
        - year: 관심 연도
        - until:
            * True: 당해를 포함한 이전 연도 모두
            * False: 당해만
    '''
    if until: df_year = df[df["year"] <= year]
    else: df_year = df[df["year"] == year]
    
    scopus_ids = list(df_year.index)
    if column == "keywords": keywords = [k.lower().split("; ") for k in df_year[column].values]
    else: keywords = [preprocess_string(k, filters=[strip_tags, strip_punctuation,strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]) for k in df_year[column].values]
        
    return scopus_ids, keywords

def measuring_novelties(docvecs, neighbors=10, method="LOF"):
    '''
        Document 벡터들을 입력했을 때, 각 벡터에 대한 novelty 점수를 반환함.
        
        - docvecs: novelty를 구하고 싶은 document vector들
        - neighbors: LOF에 사용할 이웃 벡터의 갯수
    '''
    
    assert method in ["LOF", "IF", "EE", "SVM"]

    if method == "LOF":
        clf = LocalOutlierFactor(n_neighbors=neighbors)
        clf.fit(docvecs)
        novelty = clf.negative_outlier_factor_ * -1
    elif method == "IF":
        clf = IsolationForest()
        clf.fit(docvecs)
        novelty = clf.score_samples(docvecs) * -1
    elif method == "SVM":
        clf = OneClassSVM(gamma='auto')
        clf.fit(docvecs)
        novelty = clf.score_samples(docvecs) * -1
        
    return novelty

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self, model_dir, path_prefix):
        self.avg_distances = []
        self.path_prefix = path_prefix
        self.model_dir = model_dir
        self.best = False
        self.epoch = 0
        
    # def on_train_begin(self, model):
    #     print("Train start")
    #     avg_distance = keywords2avg_distance(model, self.keywords)
    #     self.avg_distances.append(avg_distance)

    def on_epoch_end(self, model):
        # avg_distance = keywords2avg_distance(model, self.keywords)
        # self.avg_distances.append(avg_distance)
        # print(self.avg_distances)
        # 
        # if avg_distance < avg_distance[-1] and not self.best:
        #     print("Best model saving..")
        #     model.save(f'{self.model_dir}/{self.path_prefix}_epoch{self.epoch}_best.model')
        #     docvecs = np.array([keywords2vec(model, keyword) for keyword in self.keywords])
        #     np.save(f"{self.model_dir}/docvecs{epochs}.npy", docvecs)
        #     self.best = True
        self.epoch += 1
    
    # def on_train_end(self, model):
    #     print("Last model saving..")
    #     model.save(f'{self.model_dir}/{self.path_prefix}_epoch{self.epoch}.model')
    #     docvecs = np.array([keywords2vec(model, keyword) for keyword in self.keywords])
    #     np.save(f"{self.model_dir}/docvecs{epochs}.npy", docvecs)
        
        
class EpochLogger(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.avg_distance_list = []

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % 20 == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
            
def further_training_by_year(df, year, pretrained_model_path="models/pretrained/wiki.en.bin", model_dir="model", column="title", epochs=14):
    print("Load pretrained model")
    model = fasttext.load_facebook_model(pretrained_model_path)
    model.min_count = 0
    model.window = 2 ** 30

    model_dir = f"{model_dir}/year={year}"
  

    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    scopus_ids, keywords = get_keywords(df, year, until=True, column=column)
    
    print("Build vocab")
    model.build_vocab(keywords, update=True)
    
    epoch_saver = EpochSaver(model_dir=model_dir, path_prefix="model")
    epoch_logger = EpochLogger()
    
    avg_distances = []
    avg_distance = keywords2avg_distance(model, keywords)
    print(f"Epoch: initial, Average distance between words: {avg_distance}")
    avg_distances.append(avg_distance)
    
    best = False
    
    for epoch in range(epochs):
        model.train(
            corpus_iterable=keywords,
            total_examples=len(keywords),
            epochs=1,
        )
        avg_distance = keywords2avg_distance(model, keywords)
        print(f"Epoch: {epoch} Average distance between words: {avg_distance}")
        
        if avg_distance < avg_distances[-1] and not best:
            print("Best model saving...")
            model.save(f'{model_dir}/best-epoch{epoch}.model')
            docvecs = np.array([keywords2vec(model, keyword) for keyword in keywords])
            np.save(f"{model_dir}/best-docvecs{epoch}.npy", docvecs)
            avg_distances.append(avg_distance)
            break
            
        avg_distances.append(avg_distance)
    
    # print("Last model saving ")
    # model.save(f'{model_dir}/epoch{epoch}.model')
    # docvecs = np.array([keywords2vec(model, keyword) for keyword in keywords])
    # np.save(f"{model_dir}/docvecs{epoch}.npy", docvecs)

    # print("MAKE Document embeddings")
    # docvecs = np.array([keywords2vec(model, keyword) for keyword in keywords])
    # np.save(f"{model_dir}/docvecs{epochs}.npy", docvecs)
    return model

PRETRAINED_MODELS_DIR = "model/pretrained"
FURTHER_TRAINED_MODELS_DIR = "model/further-trained"

for year in range(year_start, year_end+1):
    print(f"YEAR: {year}")
    further_training_by_year(papers_all_tags, year, model_dir=FURTHER_TRAINED_MODELS_DIR, column='title', epochs=200)