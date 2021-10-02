# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# my imports
import sqlite3 # eseculitooo bebesitaaa
import pandas as pd
from time import time


def create_fts_table(cursor):
    cursor.execute("DROP TABLE IF EXISTS commit_texts;")
    cursor.execute("""CREATE VIRTUAL TABLE commit_texts USING fts3(
                    commit_hash VARCHAR(40) NOT NULL, 
                    message TEXT,
                    lines_added INT,
                    lines_removed INT,
                    files INT);""")
    
    cursor.execute("""INSERT INTO commit_texts
                    SELECT COMMIT_HASH as commit_hash, COMMIT_MESSAGE AS message, SUM(gcc.LINES_ADDED) AS lines_added, SUM(gcc.LINES_REMOVED) AS lines_removed, COUNT(DISTINCT gcc.FILE) as files
                    FROM GIT_COMMITS AS gc NATURAL JOIN GIT_COMMITS_CHANGES AS gcc
                    GROUP BY gc.COMMIT_HASH;""")



def get_commits_from_issue(issue_id: str, connection: sqlite3.Connection) -> (pd.DataFrame):
    """This function returns a dataframe with the commits related to a single issue. This also extracts
    the number of lines changed.
        Input:
            - issue_key {str}: the issue key we use to match the descriptions
            - connection {sqlite3.Connection}: the connection to the database
        Output:
            - df {pandas.DataFrame}: a dataframe containing the following columns:
                 []
    """
    commits_query = f"SELECT * FROM commit_texts WHERE message MATCH '\"{issue_id}\"'"
    df = pd.read_sql(commits_query, connection)
    
    return (issue_id, df["lines_added"].sum(), df["lines_removed"].sum(), df["files"].sum())



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    version2_path = "../data/raw/td_V2.db"
    con2 = sqlite3.connect(version2_path)
    cursor = con2.cursor()

    create_fts_table(cursor)

    jira_query = """SELECT * FROM JIRA_ISSUES"""
    df = pd.read_sql(jira_query, con2)


    metrics = df["KEY"].apply(get_commits_from_issue, args = (con2,))
    
    df["lines_added"] = list(map(lambda x: x[1], metrics))
    df["lines_removed"] = list(map(lambda x: x[2], metrics))
    df["files_changed"] = list(map(lambda x: x[3], metrics))

    df.to_csv("./data/interim/issues_with_metrics.csv")





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
