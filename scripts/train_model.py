import time
from duq_330_2025.patent_reader import read_patentees_sql
from duq_330_2025.reusable_classifier import ReusableClassifier
from name_distance import pair_up,name_distance
import sqlite3
import pandas as pd
import random
import duq_330_2025.npi_reader as npi
import fasttext
import networkx as nx


def typo_maker(name:str,delta:float,set_seed=None) -> str:
    random.seed(set_seed)
    delta = min(1,max(0,delta))
    letters = """qwertyuiop[]{}asdfghjkl;:'"zxcvbnm,<.>/? """
    num_char_change = int(delta*len(name))
    indexes = random.choices(range(len(name)),k=num_char_change)
    name = list(name)
    for i in indexes:
        current_letter = name[i]
        rand_letter = random.choice(letters)
        rand_letter = random.choice((rand_letter,rand_letter.upper()))
        while rand_letter.lower() == current_letter:
            rand_letter = random.choice(letters)
            rand_letter = random.choice((rand_letter,rand_letter.upper()))
        name[i] = rand_letter
    return ''.join(name)

def create_training_data(first_df:pd.DataFrame,
                         first_id_col:str,
                         second_df:pd.DataFrame,second_id_col:str,
                         manual_training_df:pd.DataFrame,
                         sample_num:int=60):
    """takes in two data frames of data and a third manualy selected data frame and returns a
    data frame with paired names

    Args:
        first_df (pd.DataFrame): a data frame with gathered data (has names column)
        
        second_df (pd.DataFrame): can be the same frame as first (has names column)
        
        manual_training_df (pd.DataFrame): manually checked data desired to be 
        added to training (has names column)
    """

    assert 'surname' in first_df
    assert 'forename' in first_df
    assert 'address' in first_df
    assert first_id_col in first_df
    assert 'surname' in second_df
    assert 'forename' in second_df
    assert 'address' in second_df
    assert second_id_col in second_df

    assert 'surname_a' in manual_training_df
    assert 'surname_b' in manual_training_df
    assert 'forename_a' in manual_training_df
    assert 'forename_b' in manual_training_df
    assert 'address_a' in manual_training_df
    assert 'address_b' in manual_training_df
    assert 'is_same' in manual_training_df

    first_sample_df = first_df.sample(sample_num//2,replace=False)
    second_sample_df = second_df.sample(sample_num//2,replace=False)

    first_sample_df['id'] = first_sample_df[first_id_col]
    second_sample_df['id'] = second_sample_df[second_id_col]

    sample_df = pd.concat([first_sample_df, second_sample_df])

    # manual training data
    training_df = manual_training_df
    print(len(training_df))

    # pair up same names
    training_df = pd.concat([training_df,
                             pd.DataFrame([{'surname_a':row['surname'],
                                            'surname_b':row['surname'],
                                            'forename_a':row['forename'],
                                            'forename_b':row['forename'],
                                            'address_a':row['address'],
                                            'address_b':row['address'],
                                            'is_same':1} 
                                            for _,row in sample_df.iterrows()
                                            ])
                            ])
    
    # adds the generated typos such that 
    # same name matched + typos is 70% of the the total training data
    num_gen = int((.7*sample_num-1)/.3)

    # now give them typos that make it slightly different
    for i in range(1,num_gen):
        training_df = pd.concat([training_df,
                                 pd.DataFrame([{'surname_a':row['surname'],
                                                'surname_b':typo_maker(row['surname'],i/(10*num_gen)),
                                                'forename_a':row['forename'],
                                                'forename_b':typo_maker(row['forename'],i/(10*num_gen)),
                                                'address_a':row['address'],
                                                'address_b':typo_maker(row['address'],i/(10*num_gen)),
                                                'is_same':1} 
                                                for _,row in sample_df.iterrows()])
                                                ])
    print(len(training_df))


    # give mismatch names 
    training_df = pd.concat([training_df,
                             pd.DataFrame([{'surname_a':row_a['surname'],
                                            'surname_b':row_b['surname'],
                                            'forename_a':row_a['forename'],
                                            'forename_b':row_b['forename'],
                                            'address_a':row_a['address'],
                                            'address_b':row_b['address'],
                                            'is_same':0} 
                                            for id_a,row_a in sample_df.iterrows() 
                                            for id_b,row_b in sample_df.iterrows()
                                            if id_a != id_b
                                            ])
                            ])
    print(len(training_df))
    return training_df

def train_model(features_df:pd.DataFrame,labels_series:pd.Series) -> ReusableClassifier:
    # all wrong
    """takes in a column of strings from a Dataframe and a test fraction
    and and trains an xgboost classifier with different typos distances to find
    if names are the same

    Args:
        sample_series (pd.Series): a column of strings (names)
        test_frac (float, optional): porportion of data used to test model. Defaults to .1.

    Returns:
        ReusableClassifier: a xgboost classifier trained on string distances
    """
    
    model = ReusableClassifier('xgboost')
    model.train(features_df,labels_series)
    return model

if __name__ == '__main__':
    # read in all the dataframes
    patentee_df = read_patentees_sql('data/patent_npi_db.sqlite')
    doctor_df = npi.read("data/npidata_pfile_20250303-20250309.csv")
    manual_training = pd.read_csv('data/manual_training.csv')
    # manual_training = pd.DataFrame([{'forename_a':'',
    #                                  'forename_b':'',
    #                                  'surname_a':'',
    #                                  'surname_b':'',
    #                                  'address_a':'',
    #                                  'address_b':'',
    #                                  'is_same':1}])


    training_df = create_training_data(patentee_df,'patent_id',doctor_df,'npi',manual_training,200)

    # calculate features
    training_df = name_distance(training_df)

    # train model
    model = train_model(training_df[['surname_distance','forename_distance','address_distance']],training_df['is_same'])
    

    # already loaded databases and models
    ft_model = fasttext.load_model('data/cc.en.50.bin')
    
    # blocking
    start = time.time()
    paired_test = pair_up(patentee_df,'patent_id',doctor_df,'npi',ft_model)
    print(time.time()-start)

    # calculate features
    start = time.time()
    paired_test = name_distance(paired_test)
    print(time.time()-start)

    # save the pairing and features
    paired_test.to_csv('data/paired_patentee_doctor_names.csv',index=None)

    # read in saved loading, blocking, training, and calc features
    # paired_test = pd.read_csv('data/paired_patentee_doctor_names.csv',index_col=None)

    # predict from model
    paired_test['predict'] = model.predict(paired_test[['surname_distance','forename_distance','address_distance']])
    paired_test['predict_p'] = model.predict_proba(paired_test[['surname_distance','forename_distance','address_distance']])

    # rename for sql saving
    mapper = {
        'surname_a':'patentee_surname',
        'forename_a':'patentee_forename',
        'address_a':'patentee_address',
        'id_a':'patentee_id',
        'surname_b':'doctor_surname',
        'forename_b':'doctor_forename',
        'address_b':'doctor_address',
        'id_b':'doctor_id'
    }

    sql_df = paired_test.loc[paired_test['predict'] == 1].rename(columns=mapper)

    # connected components
    edges = [(row['patentee_id'],row['doctor_id']) for _,row in sql_df.iterrows()] 
    nodes = [
             (row['patentee_id'],row['patentee_forename']+' '+row['patentee_surname']) 
             for _,row in sql_df.iterrows()
             ] + [
             (row['doctor_id'],row['doctor_forename']+' '+row['doctor_surname']) 
             for _,row in sql_df.iterrows()]
    people_graph = nx.Graph()
    people_graph.add_nodes_from(nodes)
    people_graph.add_edges_from(edges)
    connected_people = list(nx.connected_components(people_graph))

    print(set(paired_test['id_a'].tolist()).intersection(set(paired_test['id_b'].tolist())))

    # save results 
    conn = sqlite3.connect('data/patent_npi_db.sqlite')
    sql_df.to_sql('patentees_and_doctors',conn,if_exists='append',index=False)
    conn.close()
    
    divided_dfs:list[pd.DataFrame] = []
    for i in range(10):
        divided_dfs.append(paired_test.loc[(i/10 <= paired_test['predict_p']) & 
                                           ((i+1)/10 > paired_test['predict_p'])])
    
    print(len(divided_dfs))
    print(list(map(len,divided_dfs)))
    manual_dicts = []
    for bin in divided_dfs:
        bin = bin.sample(min(20,len(bin)))
        for _,entry in bin.iterrows():
            print(entry['forename_a']+' '+entry['surname_b']+' | '+entry['forename_b']+' '+entry['surname_b'],entry['address_a']+' | '+entry['address_b'],sep='\n')
            manual_pick = input('is same: ')
            manual_dicts.append( {'forename_a':entry['forename_a'],
                                  'forename_b':entry['forename_b'],
                                  'surname_a':entry['surname_a'],
                                  'surname_b':entry['surname_b'],
                                  'address_a':entry['address_a'],
                                  'address_b':entry['address_b'],
                                  'id_a':entry['id_a'],
                                  'id_b':entry['id_b'],
                                  'is_same':int(manual_pick)} )
    manual_training = pd.concat([manual_training,pd.DataFrame(manual_dicts)],ignore_index=True)
    manual_training.to_csv('data/manual_training.csv',index=None)

    print('kk')