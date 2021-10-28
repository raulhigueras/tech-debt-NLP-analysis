# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# general imports
import numpy as np
import pandas as pd
import pickle 

# topic modeling imports
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.test.utils import datapath

# classification model imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack



def train_topic_model(data_src):
    """Trains a LDA model of 11 topics. See topic-modelling notebook for more info"""
    
    df = pd.read_csv(data_src, index_col=0)
    data_words = [str(txt).split(" ") for txt in df['text']]
    id2word = corpora.Dictionary(data_words)

    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]

    num_topics = 11
    lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=12345)

    return ([id2word, lda_model], ['id2word.pickle', 'lda11.pickle'])


def train_classification_model(data_src):
    """Trains a MLP classification model. See classification notebook for more info"""

    df_raw = pd.read_csv(data_src, index_col=0)

    df = df_raw[(df_raw["num_commits"]>0) & (df_raw["num_commits"]<3e5)].dropna()
    good_projects = [proj for proj in set(df['project_id']) if sum(df['project_id'] == proj) > 500]
    df = df[df['project_id'].isin(good_projects)].reset_index(drop=True)

    varname = 'lines_added'
    limits = np.array([0, 15, 118, 3872353])
    def get_target(x, limits):
        return np.where(x >= limits)[0][-1]

    y = df[varname].apply(get_target, args=(limits,))
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df["text"])
    svd = TruncatedSVD(n_components=500, random_state=123)
    X_svd = svd.fit_transform(X_tfidf)

    enc = OneHotEncoder()
    enc.fit(np.asarray(df['type']).reshape(-1,1))

    X_onehot = enc.transform(np.asarray(df['type']).reshape(-1,1))
    X = hstack((X_svd, X_onehot))

    mlp = MLPClassifier(hidden_layer_sizes=(15,10,5), max_iter=500, activation='relu', early_stopping=True, warm_start=True, random_state=123).fit(X, y)

    return [(vectorizer, svd, enc, mlp),
            ('vect.pickle', 'svd.pickle', 'onehot.pickle', 'mlp.pickle'), mlp.score(X,y)]


@click.command()
@click.argument('data_src', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(data_src, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Training all the models...')

    # Training the topic model
    logger.info('Training LDA for Topic Modeling')
    mods, names = train_topic_model(data_src)
    for name, mod in zip(names, mods):
        fp = f'{output_filepath}/{name}'
        if name == "lda11.pickle":
            mod.save("models/lda11")
        else:
            pickle.dump(mod, open(fp, 'wb'))
    logger.info('Done! :)')

    # Training the classification model
    logger.info('Training MLP for Difficulty Classification')
    mods, names, acc = train_classification_model(data_src)
    for name, mod in zip(names, mods):
        fp = f'{output_filepath}/{name}'
        pickle.dump(mod, open(fp, 'wb'))
    logger.info(f'Done! :) -- Training accuracy: {acc}')

    logger.info(f'All models saved in {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
