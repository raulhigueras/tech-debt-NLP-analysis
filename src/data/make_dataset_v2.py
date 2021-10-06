# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# my imports
import sqlite3 # eseculitooo bebesitaaa
import pandas as pd
from time import time



def get_insert(version):
    ins1 = """INSERT INTO commit_texts
                    SELECT commitHash as commit_hash, commitMessage AS message, SUM(gcc.linesAdded) AS lines_added, SUM(gcc.linesRemoved) AS lines_removed, COUNT(DISTINCT gcc.newPath) as files
                    FROM GIT_COMMITS AS gc NATURAL JOIN GIT_COMMITS_CHANGES AS gcc
                    GROUP BY gc.commitHash;"""

    ins2 = """INSERT INTO commit_texts
                    SELECT COMMIT_HASH as commit_hash, COMMIT_MESSAGE AS message, SUM(gcc.LINES_ADDED) AS lines_added, SUM(gcc.LINES_REMOVED) AS lines_removed, COUNT(DISTINCT gcc.FILE) as files
                    FROM GIT_COMMITS AS gc NATURAL JOIN GIT_COMMITS_CHANGES AS gcc
                    GROUP BY gc.COMMIT_HASH;"""

    if version == 1:
        return ins1
    elif version == 2:
        return ins2
    
    raise ValueError("The version number supported is 1 or 2.")


def create_fts_table(cursor, version=2):
    # the creation of the table is always the same regardless of the version
    cursor.execute("DROP TABLE IF EXISTS commit_texts;")
    cursor.execute("""CREATE VIRTUAL TABLE commit_texts USING fts3(
                    commit_hash VARCHAR(40) NOT NULL, 
                    message TEXT,
                    lines_added INT,
                    lines_removed INT,
                    files INT);""")
    
    insert_text = get_insert(version)
    cursor.execute(insert_text)



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


column_renamer = {
    # from jira_issues:
    'projectID': "project_id",
    'creationDate': "creation_date",
    'resolutionDate': "resolution_date",
}

def read_sql_table_from_connection(query, connection, columns):
    # read and rename columns
    df = pd.read_csv(query, connection).rename(columns = column_renamer)
    # all the names in lowercase
    df.columns = list(map(lambda x: x.lower(), df.columns))

    return df[columns]


def add_changes_metrics(df, connection):
    out_df = df.copy()
    metrics = df["key"].apply(get_commits_from_issue, args = (connection,))   
    
    out_df["lines_added"] = list(map(lambda x: x[1], metrics))
    out_df["lines_removed"] = list(map(lambda x: x[2], metrics))
    out_df["files_changed"] = list(map(lambda x: x[3], metrics))

    return out_df


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # connection and cursor of version 1
    version1_path = "../../data/raw/td_V1.db"
    con1 = sqlite3.connect(version1_path)
    cursor1 = con1.cursor()

    # connection and cursor of version 2
    version2_path = "../../data/raw/td_V2.db"
    con2 = sqlite3.connect(version2_path)
    cursor2 = con2.cursor()

    create_fts_table(cursor1, 1)
    create_fts_table(cursor2, 2)

    columns = ["key", "project_id", "creation_date", "resolution_date", "summary", "description", "type", 
                "lines_added", "lines_removed", "files"]
    jira_query = """SELECT * FROM JIRA_ISSUES"""
    jira1 = read_sql_table_from_connection(jira_query, con1, columns)
    jira2 = read_sql_table_from_connection(jira_query, con2, columns)

    # filter the version 1 removing the issues in 2
    issues_from1 = list(set(jira1["key"]) - set(jira2["key"]))
    jira1 = jira1[jira1["key"].isin(issues_from1)]

    full_df1 = add_changes_metrics(jira1, con1)
    full_df2 = add_changes_metrics(jira2, con2)

    final = pd.merge(full_df1, full_df2, how = "outer")

    final.to_csv("../../data/interim/issues_with_metrics.csv")





if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
