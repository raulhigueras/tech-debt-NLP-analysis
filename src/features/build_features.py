# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# my imports
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


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


def removeProjectSpecificTerms(txts, project_names):
    project_names_re = r'|'.join(project_names)
    txts = txts.apply(lambda txt: re.sub(project_names_re, ' ', str(txt)))

    return txts


@click.command()
@click.argument('data_src', type=click.Path(exists=True))
@click.argument('interim_filepath', type=click.Path())
def main(data_src, interim_filepath):

    logger = logging.getLogger(__name__)

    logger.info("Reading processed dataset...")
    df = pd.read_csv(data_src, index_col=0)
    df.dropna(subset=["project_id"], inplace=True)
    df['description'] = df['description'].fillna('')

    logger.info("Applying NLP text preprocessing methods...")
    pnames = {" " + name.split(":")[-1] + " "
              for name in set(df['project_id'])}

    df['summary'] = unifyStyle(df['summary'])
    df['description'] = unifyStyle(df['description'])

    inter_text = df['summary'] + ' ' + df['description']

    df['summary'] = removeStopwordsandStemmer(df['summary'])
    df['description'] = removeStopwordsandStemmer(df['description'])

    df['summary'] = removeProjectSpecificTerms(df['summary'], pnames)
    df['description'] = removeProjectSpecificTerms(df['description'], pnames)

    df['text'] = df['summary'] + ' ' + df['description']
    df['inter_text'] = inter_text
    df = df.drop(columns=['summary', 'description'])

    logger.info("Done! Saving intermediate dataset with preprocessed text")
    df.to_csv(f"{interim_filepath}/preproc.csv")
    logger.info(f"Saved in {interim_filepath}/preproc.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
