# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# my imports
import os
import shutil as sh
import sqlite3
import pandas as pd
import datetime as dt
import requests
import zipfile


VERSION1 = "tdv1.db"
VERSION2 = "tdv2.db"


def download_data_from_github(folder_path):
    """This fucntion checks whether the .db files are in data/raw;
    if they are, it does nothing; else, it downloads the data from github.
    The downloading process can take a while.
    """
    raw_files = os.listdir(folder_path)
    # if we already have the files, don't download
    if VERSION1 in raw_files and VERSION2 in raw_files:
        return
    # else, download the data from github.
    link_v1 = "https://github.com/clowee/The-Technical-Debt-Dataset/releases/download/1.0.1/TechnicalDebtDataset_v1.01.db.zip"
    link_v2 = "https://github.com/clowee/The-Technical-Debt-Dataset/releases/download/2.0/td_V2.db"

    # get the .zip file of the v1
    r = requests.get(link_v1)
    with open(f"{folder_path}/{VERSION1}.zip", "wb") as f:
        f.write(r.content)

    # extract the database from the .zip file and remove the old zip.
    with zipfile.ZipFile(f"{folder_path}/{VERSION1}.zip", 'r') as zip_ref:
        files_in = zip_ref.namelist()
        print(files_in)
        my_file = [x for x in files_in if ".db" in x]
        if len(my_file) > 0:
            name_of_file = my_file[0]

        zip_ref.extract(name_of_file, f"{folder_path}/data")
    
    sh.move(f"{folder_path}/data/*.db", f"{folder_path}/{VERSION1}")

    os.remove(f"{folder_path}/{VERSION1}.zip")
    os.remove(f"{folder_path}/data")

    # get the database of the v2
    r = requests.get(link_v2)
    with open(f"{folder_path}/{VERSION2}", "wb") as f:
        f.write(r.content)


def get_insert(version):
    """Get the insert code for the given version. This is added to deal with
       the different column names from V1 and V2."""
    ins1 = """INSERT INTO commit_texts
                SELECT commitHash as commit_hash,
                        commitMessage AS message,
                        SUM(gcc.linesAdded) AS lines_added,
                        SUM(gcc.linesRemoved) AS lines_removed,
                        COUNT(DISTINCT gcc.newPath) as files
                FROM GIT_COMMITS AS gc NATURAL JOIN GIT_COMMITS_CHANGES AS gcc
                GROUP BY gc.commitHash;"""

    ins2 = """INSERT INTO commit_texts
                SELECT COMMIT_HASH as commit_hash,
                       COMMIT_MESSAGE AS message,
                       SUM(gcc.LINES_ADDED) AS lines_added,
                       SUM(gcc.LINES_REMOVED) AS lines_removed,
                       COUNT(DISTINCT gcc.FILE) as files
                FROM GIT_COMMITS AS gc NATURAL JOIN GIT_COMMITS_CHANGES AS gcc
                GROUP BY gc.COMMIT_HASH;"""

    if version == 1:
        return ins1
    elif version == 2:
        return ins2

    raise ValueError("The version number supported is 1 or 2.")


def create_fts_table(con, version=2):
    """This function creates a FTS3 table in the database from cursor to index
    the comits by message and easily filtrate by this column.
    The process is inline.
    """
    cursor = con.cursor()
    # the creation of the table is always the same regardless of the version
    # cursor.execute("DROP TABLE IF EXISTS commit_texts;") 
    cursor.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS commit_texts USING fts3(
                    commit_hash VARCHAR(40) NOT NULL,
                    message TEXT,
                    lines_added INT,
                    lines_removed INT,
                    files INT);""")

    insert_text = get_insert(version)
    cursor.execute(insert_text)
    con.commit()


def get_commits_from_issue(issue_id: str, connection: sqlite3.Connection):
    """This function returns a dataframe with the commits related to a single
    issue. This also extracts the number of lines changed.
        Input:
            - issue_key {str}: the issue key we use to match the descriptions
            - connection {sqlite3.Connection}: the connection to the database
        Output:
            - (lines_added, lines_removed, files):
    """
    commits_query = f"""SELECT *
                        FROM commit_texts
                        WHERE message MATCH '\"{issue_id}\"'"""
    df = pd.read_sql(commits_query, connection)

    return (issue_id, df["lines_added"].sum(), df["lines_removed"].sum(),
            df["files"].sum())


# simple dictionary to rename the columns from v1 to the standard.
column_renamer = {
    # from jira_issues:
    'projectID': "project_id",
    'creationDate': "creation_date",
    'resolutionDate': "resolution_date",
}


def read_sql_table_from_connection(query, connection, selected_columns=None):
    """This function executes a SQL query in the given connection. Although you
    can specify the columns selected in the query, you can specify the columns
    needed in selected_columns. This will help to deal with different versions
    of the dataset.
    """
    # read and rename columns
    df = pd.read_sql(query, connection).rename(columns=column_renamer)
    # all the names in lowercase
    df.columns = list(map(lambda x: x.lower(), df.columns))

    if selected_columns is not None:
        return df[selected_columns]

    return df


def add_changes_metrics(df, connection):
    """This function joins the data from the jira_issues table with the data
    from the FTS3 table. It gets, for each issue, the number of lines_added,
    lines_removed and files_changed.
    """
    out_df = df.copy()
    metrics = df["key"].apply(get_commits_from_issue, args=(connection,))

    out_df["lines_added"] = list(map(lambda x: x[1], metrics))
    out_df["lines_removed"] = list(map(lambda x: x[2], metrics))
    out_df["files_changed"] = list(map(lambda x: x[3], metrics))

    return out_df


def parse_date(x):
    if type(x) == str:
        if "T" in x:
            return dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000+0000")
        elif x == "":
            return pd.NaT
        else:
            return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S +0000")
    elif type(x) == dt.datetime:
        return x
    else:
        raise ValueError("Type incorrect")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info('Checking if raw data is available, else downloading')
    download_data_from_github(input_filepath)
    logger.info('raw data is in the house!')

    # connection and cursor of version 1
    version1_path = f"{input_filepath}/{VERSION1}"
    con1 = sqlite3.connect(version1_path)

    # connection and cursor of version 2
    version2_path = f"{input_filepath}/{VERSION2}"
    con2 = sqlite3.connect(version2_path)

    create_fts_table(con1, 1)
    create_fts_table(con2, 2)

    select_columns = ["key", "project_id", "creation_date", "resolution_date",
                      "summary", "description", "type"]  # jira columns
    jira_query = """SELECT * FROM JIRA_ISSUES"""
    jira1 = read_sql_table_from_connection(jira_query, con1, select_columns)
    jira2 = read_sql_table_from_connection(jira_query, con2, select_columns)

    # filter the version 1 removing the issues in 2
    issues_from1 = list(set(jira1["key"]) - set(jira2["key"]))
    jira1 = jira1[jira1["key"].isin(issues_from1)]

    full_df1 = add_changes_metrics(jira1, con1)
    full_df2 = add_changes_metrics(jira2, con2)

    final = pd.merge(full_df1, full_df2, how="outer")

    ### COMPUTE THE DURATION ###

    finished = final[~final["resolution_date"].isna()]
    durations = finished["resolution_date"].apply(parse_date) - finished["creation_date"].apply(parse_date)
    finished["duration"] = [d.days*24 + d.seconds/3600 for d in durations]

    final = final.join(finished["duration"])

    final.to_csv(f"{output_filepath}/issues_with_metrics.csv")

    logger.info('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
