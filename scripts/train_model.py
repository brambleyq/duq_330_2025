from xgboost import train
from duq_330_2025.reusable_classifier import ReusableClassifier
from name_distance import pair_up,name_distance
import sqlite3
import pandas as pd
import random

def typo_maker(name:str,num_char_change:int) -> str:
    num_char_change = min(len(name),max(0,num_char_change))
    if num_char_change == 0:
        return name 
    letters = """qwertyuiop[]{}asdfghjkl;:'"zxcvbnm,<.>/? """
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

def train_model(sample_series:pd.Series,test_frac:float=.1) -> ReusableClassifier:
    """takes in a column of strings from a Dataframe and a test fraction
    and and trains an xgboost classifier with different typos distances to find
    if names are the same

    Args:
        sample_series (pd.Series): a column of strings (names)
        test_frac (float, optional): porportion of data used to test model. Defaults to .1.

    Returns:
        ReusableClassifier: a xgboost classifier trained on string distances
    """
    # set the initial training df to be just a pair of the same sampled names
    training_df = pd.DataFrame([[name,name,1] 
                                for name in sample_series]
                                ,columns=['names_a','names_b','is_same'])
    # now give them typos that make it slightly different
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,typo_maker(name,1),1] 
                                        for name in training_df['names_a'].unique()]
                                        ,columns=['names_a','names_b','is_same'])])
    training_df = pd.concat([training_df,
                            pd.DataFrame([[name,typo_maker(name,2),1] 
                                        for name in training_df['names_a'].unique()]
                                        ,columns=['names_a','names_b','is_same'])])
    
    # gets mismatched data
    for _ in range(3):
        scrambled_sample = sample_series.sample(frac=1, replace = True)
        training_df = pd.concat([training_df,
                                    pd.DataFrame([[sample_series.iloc[i],scrambled_sample.iloc[i],0] 
                                                for i in range(len(sample_series)) 
                                                if sample_series.iloc[i] != scrambled_sample.iloc[i]]
                                                ,columns=['names_a','names_b','is_same'])])
    # half of the name is wrong so there is no simmilarity 
    # training_df = pd.concat([training_df,
    #                         pd.DataFrame([[name,typo_maker(name,len(name)),0] 
    #                                     for name in training_df['names_a'].unique()]
    #                                     ,columns=['names_a','names_b','is_same'])])
    training_df = name_distance(training_df)
    
    model = ReusableClassifier('xgboost')
    model.train(training_df[['name_distance']],training_df['is_same'],test_frac=test_frac)
    return model
    

if __name__ == '__main__':
    conn = sqlite3.connect('data/patent_npi_db.sqlite')
    assignee_df = pd.read_sql('SELECT * FROM assignors',conn)
    conn.close()
    assignee_df['name'] = assignee_df.apply(lambda row: row['forename'] +' '+ row['surname'],axis=1)
    
    confirm_frac = .5
    split_index = int(confirm_frac*len(assignee_df))
    
    # shuffle data then split into training and confirmation 
    # (training will be split further into training and testing)
    assignee_df = assignee_df.sample(frac=1,replace=False)
    training_df = assignee_df[:split_index]
    confirmation_df = assignee_df[split_index:]

    model = train_model(training_df['name'])

    paired_df = pair_up(confirmation_df['name'],confirmation_df['name'])
    paired_df = name_distance(paired_df)
    paired_df['name_distance'].min()

    paired_df['prediction'] = model.predict(paired_df[['name_distance']])
    
    print(paired_df)
    print(paired_df['prediction'].mean())
    print('kk')
    