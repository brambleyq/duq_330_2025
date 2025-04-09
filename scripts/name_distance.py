import sqlite3
import time
import pandas as pd
from fuzzywuzzy import fuzz
import fasttext
import hnswlib
import numpy as np

from duq_330_2025 import npi_reader
from duq_330_2025.patent_reader import read_patentees_sql


# need to add a way ot keep the ids and bake it a generator


def pair_up(first_df:pd.DataFrame,first_id_col:str,second_df:pd.DataFrame,second_id_col:str,fastText:fasttext.FastText._FastText,nearest_neighbors:int = 1000) -> pd.DataFrame:
    """takes in two dataframes and return a paired names dataframe

    Args:
        first_df (pd.DataFrame): first data frame should be the smaller one
        
        first_id_col (str): the column that contains the first data frames id
        
        second_df (pd.DataFrame): second df, can be same as first
        
        second_id_col (str): the column that contains the seconds data frames id
        
        fastText (fasttest.FastText._FastText): a fast text model (the 50d pretrained one)
        
        nearest_neighbors (int): the number of nearest neighbors wanted to pair 
        the second data frame with

    Returns:
        pd.DataFrame: a data frame with columns: names_a, names_b
        with len = len(names)*nearest_neighbors
    """

    assert 'address' in first_df
    assert 'forename' in first_df
    assert 'surname' in first_df
    assert first_id_col in first_df
    assert 'address' in second_df
    assert 'forename' in second_df
    assert 'surname' in second_df
    assert second_id_col in second_df
    

    first_df['forename_vec'] = first_df['forename'].apply(fastText.get_sentence_vector)
    second_df['forename_vec'] = second_df['forename'].apply(fastText.get_sentence_vector)
    
    pairs = []
    second_vectors = np.array([array.tolist() for array in second_df['forename_vec']])

    p = hnswlib.Index(space='l2',dim=50)
    p.init_index(len(first_df)+len(second_df))
    p.add_items(second_vectors,second_df.index)

    for _,row_a in first_df.iterrows():
        closest_indexes,_ = p.knn_query(row_a['forename_vec'],nearest_neighbors)

        closest_df = second_df.loc[closest_indexes.flatten()]
        pairs += [{'id_a':row_a[first_id_col],
                   'forename_a':row_a['forename'],
                   'surname_a':row_a['surname'],
                   'address_a':row_a['address'],
                   'id_b':row_b[second_id_col],
                   'forename_b':row_b['forename'],
                   'surname_b':row_b['surname'],
                   'address_b':row_b['address'],} 
                   for _,row_b in closest_df.iterrows()]
    return pd.DataFrame(pairs)

def name_distance(paired_dataset:pd.DataFrame) -> pd.DataFrame:
    """takes in a paired data frame and returns that data frame
    with the distances of the pair of names

    Args:
        paired_dataset (pd.DataFrame): a data frame with columns 
        surname_a, surname_b, forename_a, forename_b, address_a, address_b

    Returns:
        pd.DataFrame: adds a columns for the 
        distances surname_distance and forename_distance and address_distance
    """
    assert 'surname_a' in paired_dataset
    assert 'surname_b' in paired_dataset
    assert 'forename_a' in paired_dataset
    assert 'forename_b' in paired_dataset
    assert 'address_a' in paired_dataset
    assert 'address_b' in paired_dataset

    paired_dataset['surname_distance'] = paired_dataset.apply(
        lambda row:fuzz.ratio(row['surname_a'],row['surname_b']),
        axis=1)
    paired_dataset['forename_distance'] = paired_dataset.apply(
        lambda row:fuzz.ratio(row['forename_a'],row['forename_b']),
        axis=1)
    paired_dataset['address_distance'] = paired_dataset.apply(
        lambda row:fuzz.ratio(row['address_a'],row['address_b']),
        axis=1)
    return paired_dataset

if __name__ == '__main__':
    assignee_df = read_patentees_sql('data/patent_npi_db.sqlite')
    doctor_df = npi_reader.read('data/npidata_pfile_20250303-20250309.csv')
    
    ft_model = fasttext.load_model('data/cc.en.50.bin')
    
    start = time.time()
    paired = pair_up(assignee_df,'patent_id',doctor_df,'npi',ft_model)
    print(time.time()-start)
    # took a little under 2.5 hours
    # now only like 30 mins
    start = time.time()
    paired = name_distance(paired)
    print(time.time()-start)

    paired.to_csv('data/paired_assignee_npi_names.csv',index=None)
    print('kk')