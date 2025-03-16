import sqlite3
import pandas as pd
from fuzzywuzzy import fuzz

def pair_up(column1:pd.Series,column2:pd.Series) -> pd.DataFrame:
    """takes in two collumns and return the every pair of values
    in those columns as a data frame

    Args:
        column1 (pd.Series): first column
        column2 (pd.Series): second column, can be the same as first

    Returns:
        pd.DataFrame: a data frame with columns names_a from first column
        and names_b from second column
    """
    return pd.DataFrame([[val1,val2] for val1 in column1 for val2 in column2],
                         columns=['names_a','names_b'])

def name_distance(paired_dataset:pd.DataFrame) -> pd.DataFrame:
    """takes in a paired data frame and returns that data frame
    with the distances of the pair of names

    Args:
        paired_dataset (pd.DataFrame): a data frame with columns names_a and names_b

    Returns:
        pd.DataFrame: adds a column name_distance which is the distance from names_a to names_b
    """
    paired_dataset['name_distance'] = paired_dataset.apply(
        lambda row:fuzz.ratio(row['names_a'],row['names_b']),
        axis=1)
    return paired_dataset

if __name__ == '__main__':
    conn = sqlite3.connect('data/patent_npi_db.sqlite')
    assignee_df = pd.read_sql('SELECT * FROM assignors',conn)
    conn.close()
    assignee_df['name'] = assignee_df.apply(lambda row: row['forename'] +' '+ row['surname'],axis=1)
    print(len(assignee_df))
    paired = pair_up(assignee_df['name'],assignee_df['name'])
    print(name_distance(paired))
    print('kk')