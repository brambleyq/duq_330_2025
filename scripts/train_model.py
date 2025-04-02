import time
from duq_330_2025.patent_reader import read_assignees_sql
from duq_330_2025.reusable_classifier import ReusableClassifier
from name_distance import pair_up,name_distance
import sqlite3
import pandas as pd
import random
import duq_330_2025.npi_reader as npi
import fasttext

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

def create_training_data(first_df:pd.DataFrame,second_df:pd.DataFrame,manual_training_df:pd.DataFrame):
    """takes in two data frames of data and a third manualy selected data frame and returns a
    data frame with paired names

    Args:
        first_df (pd.DataFrame): a data frame with gathered data (has names column)
        second_df (pd.DataFrame): can be the same frame as first (has names column)
        manual_training_df (pd.DataFrame): manually checked data desired to be added to training (has names column)
    """
    sample_df = pd.concat([first_df.sample(10), second_df.sample(10)])
    # pair up same names
    training_df = manual_training_df
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,name,1] 
                                        for name in sample_df['name']]
                                        ,columns=['names_a','names_b','is_same'])])
    # now give them typos that make it slightly different
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,typo_maker(name,.05),1] 
                                        for name in sample_df['name']]
                                        ,columns=['names_a','names_b','is_same'])])
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,typo_maker(name,.1),2] 
                                        for name in sample_df['name']]
                                        ,columns=['names_a','names_b','is_same'])])
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,typo_maker(name,.15),1] 
                                        for name in sample_df['name']]
                                        ,columns=['names_a','names_b','is_same'])])
    
    for _ in range(4):
        scrambled_sample = sample_df.sample(frac=1, replace = True)
        training_df = pd.concat([training_df,
                                    pd.DataFrame([[scrambled_sample['name'].iloc[i],scrambled_sample['name'].iloc[i],0] 
                                                for i in range(len(scrambled_sample)) 
                                                if (scrambled_sample['name'].iloc[i]) != (scrambled_sample['name'].iloc[i])]
                                                ,columns=['names_a','names_b','is_same'])])

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

def read_manual_training(path:str) -> pd.DataFrame:
    return pd.read_csv(path)[['names_a','names_b','is_same']]

if __name__ == '__main__':
    assignee_df = read_assignees_sql('data/patent_npi_db.sqlite')
    npi_df = npi.read("data/npidata_pfile_20250303-20250309.csv")
    
    manual_training = pd.read_csv('data/manual_training.csv')
    training_df = create_training_data(assignee_df,assignee_df,manual_training)

    training_df = name_distance(training_df)

    model = train_model(training_df[['name_distance']],training_df['is_same'])
    ft_model = fasttext.load_model('data/cc.en.50.bin')
    start = time.time()
    paired_test = pd.read_csv('data/paired_assignee_npi_names.csv',index_col=0)
    print(time.time()-start)

    paired_test['predict'] = model.predict(paired_test[['name_distance']])
    paired_test['predict_p'] = model.predict_proba(paired_test[['name_distance']])
    
    conn = sqlite3.connect('data/patent_npi_db.sqlite')
    paired_test.to_sql('patentees_and_doctors',conn,if_exists='append')
    conn.close()

    print(paired_test['predict'].sum())
    print(paired_test.loc[paired_test['predict'] > 0])
    
    paired_test.to_csv('data/paired_assignee_npi_names.csv',index=None)
    divided_dfs:list[pd.DataFrame] = []
    for i in range(10):
        divided_dfs.append(paired_test.loc[(i/10 <= paired_test['predict_p']) & ((i+1)/10 > paired_test['predict_p'])])
    
    print(len(divided_dfs))
    print(list(map(len,divided_dfs)))
    manual_dicts = []
    for bin in divided_dfs:
        bin = bin.sample(min(10,len(bin)))
        for ind,entry in bin.iterrows():
            print(entry)
            manual_pick = input('is same: ')
            manual_dicts.append( {'names_a':entry['names_a'],
                            'names_b':entry['names_b'],
                            'is_same':int(manual_pick)} )
    manual_training = pd.concat([manual_training,pd.DataFrame(manual_dicts)],ignore_index=True)
    manual_training.to_csv('data/manual_training.csv',index=None)

    print('kk')