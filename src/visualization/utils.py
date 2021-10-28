import numpy as np
import pandas as pd

import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel


id_to_topic = {0: "🔧 Implementation", 1: "🐘 hadoop", 2: "📞 support",
               3: "📊 database", 4: "🔧 implementation", 5: "🔄 distribution",
               6: "💾 memory", 7: "📂 compilation", 8: "📶 connection",
               9: "📊 database", 10: "📶 connection"}

id_to_commonness = {0: "⭐️⭐️⭐️Very common", 1: "⭐️⭐️Common", 2: "⭐️⭐️Common",
                    3: "⭐️⭐️Common", 4: "⭐️Rare", 5: "⭐️Rare", 6: "⭐️Rare",
                    7: "⭐️Rare", 8: "⭐️Rare", 9: "⭐️Rare", 10: "⭐️Rare"}

class_to_difficulty = {0: "❗️Easy", 1: "❗️❗️Medium", 2: "❗️❗️❗️Hard"}


def unifyStyle(txts):
    txts = txts.apply(lambda txt: str(txt).lower())

    brackets_re = r'\[+(.*?)\]+|\{+(.*?)\}+'
    URL_re = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]\
        {1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    PATH_re = r'(\/)*[\w\-\_.\<\>]+(\/[\w\-\_.\<\>\=]+)+'
    version_re = r'\d+(\.\d+)+'
    file_re = r'[\w\-\_]+(\.[\w\-\_\=]+)+'
    date_re = r'(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct\
                |Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|\
                Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)\
                ?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\
                \d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|\
                [3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?\
                [1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|\
                (?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})'
    punctuation_re = r'[^\w\s]'
    multiplespace_re = r'  +'

    txts = txts.apply(lambda txt: txt.replace('\r', ' '))
    txts = txts.apply(lambda txt: txt.replace('\n', ' '))
    txts = txts.apply(lambda txt: re.sub(brackets_re, '', txt))
    txts = txts.apply(lambda txt: re.sub(URL_re, 'URL', txt))
    txts = txts.apply(lambda txt: re.sub(PATH_re, ' ', txt))
    txts = txts.apply(lambda txt: re.sub(version_re, ' ', txt))
    txts = txts.apply(lambda txt: re.sub(file_re, ' ', txt))
    txts = txts.apply(lambda txt: re.sub(date_re, 'DATE', txt))
    txts = txts.apply(lambda txt: re.sub(punctuation_re, ' ', txt))
    txts = txts.apply(lambda txt: re.sub(multiplespace_re, ' ', txt))

    return txts


def removeStopwordsandStemmer(txts):
    words = [word_tokenize(txt) for txt in txts]

    sw = stopwords.words('english')
    words_wo_sw = [[word for word in txt if word not in sw] for txt in words]
    words_wo_sw = [[word for word in txt if not word.isnumeric()]
                   for txt in words_wo_sw]

    lemmatizer = WordNetLemmatizer()
    words_wo_sw = [[lemmatizer.lemmatize(w) for w in txt]
                   for txt in words_wo_sw]

    txts = pd.Series([' '.join(txt) for txt in words_wo_sw])

    return txts


def raw_text_to_df(text):

    text = pd.Series(text)

    text = unifyStyle(text)

    text = removeStopwordsandStemmer(text)

    df_txt = pd.DataFrame()
    df_txt["text"] = text

    return df_txt


def text_to_svd_vector(text):
    # load tfidf
    tfidf = pickle.load(open("../../models/tfidf.pkl", 'rb'))

    vectorized = tfidf.transform(text['text'])

    # load svd
    svd = pickle.load(open("../../models/svd.pkl", "rb"))

    truncated = svd.transform(vectorized)

    return truncated


def svd_vector_to_difficulty(embedding, issue_type):
    # load one hot encoding model
    enc = pickle.load(open("../../models/onehotenc.pkl", 'rb'))

    types_one_hot = enc.transform(
        np.asarray(pd.Series(issue_type)).reshape(-1, 1)).todense()
    data = np.hstack((embedding, types_one_hot))

    # load logistic regressor
    model = pickle.load(open("../../models/classifier3.pkl", 'rb'))

    predicted_class = model.predict(data)

    return class_to_difficulty[predicted_class[0]]


def text_to_tf(text):
    # load tf
    id2word = pickle.load(open("../../models/idword.pkl", 'rb'))

    return id2word.doc2bow(text['text'].values[0].split(" "))


def bow_to_topic(embedding):
    # load lda
    lda = LdaModel.load("../../models/lda_model_files/lda.model")

    # topic probability distribution
    topic_distribution = lda[embedding]

    return id_to_topic[max(topic_distribution, key=lambda x: x[1])[0]],\
        id_to_commonness[max(topic_distribution, key=lambda x: x[1])[0]]
