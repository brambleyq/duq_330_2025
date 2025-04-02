import sqlite3
import time
import pandas as pd
from fuzzywuzzy import fuzz
import fasttext
from scipy.spatial import KDTree
import numpy as np

from duq_330_2025 import npi_reader


def pair_up(first_df:pd.DataFrame,second_df:pd.DataFrame,fastText:fasttext.FastText._FastText) -> pd.DataFrame:
    """takes in two dataframes and return a paired names dataframe

    Args:
        first_df (pd.DataFrame): first df
        second_df (pd.DataFrame): second df, can be same as first
        fastText (fasttest.FastText._FastText): a fast text model (the 50d pretrained one)

    Returns:
        pd.DataFrame: a data frame with columns: names_a, names_b
        with len: len(names)*1000
    """


    first_df['name_vec'] = first_df['name'].apply(fastText.get_sentence_vector)
    second_df['name_vec'] = second_df['name'].apply(fastText.get_sentence_vector)
    
    pairs = []
    second_vectors = np.concatenate([[array] for array in second_df['name_vec']],axis=0)
    kdtree = KDTree(second_vectors)

    for _,row in first_df.iterrows():
        _,closest_indexes = kdtree.query(row['name_vec'],1000)
        closest = second_vectors[closest_indexes]
        closest_df = second_df.loc[second_df['name_vec'].apply(lambda vec: vec in closest)]
        name_a = row['name']
        pairs += [[name_a,name_b] for name_b in closest_df['name']]
    return pd.DataFrame(pairs,columns=['names_a','names_b'])

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

    npi_df = npi_reader.read('data/npidata_pfile_20250303-20250309.csv')
    
    ft_model = fasttext.load_model('data/cc.en.50.bin')
    assignee_df['name'] = assignee_df.apply(lambda row: row['forename'] +' '+ row['surname'],axis=1)
    print(len(assignee_df))
    print(len(npi_df))
    start = time.time()
    paired = pair_up(assignee_df,npi_df,ft_model)
    print(time.time()-start)
    # took a little under 2.5 hours
    print(name_distance(paired))
    print('kk')