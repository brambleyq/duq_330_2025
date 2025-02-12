"""Example of classes that we already know
    OOP is programing with "objects:. Objects are 
    nothing other than a single copy of a class with 
    specific data inside, the single copy is called 
    an 'instance' of the class
    
    a = 'Duquesne'
    a.find('Duq')

    class String:
        # Innit is the first funciton run and 
        # is hard to name a funtion this by accident

        def __init__(self):
            # self is the name of hte specific copy of hte data/class
            # otherwise known as the instance of the class
            content = ''

        def set(self, new_content):
            self.content = new_content
        
        def find(self,substring):
            # Look for substring
"""
import numpy as np
import pandas as pd
from zipfile import ZipFile

class WineQuality: 
    def __init__(self):
        # add all variables specific to the class here
        self.df:pd.DataFrame = None
        self.train = None
        self.test = None

    def read(self, path:str):
        """read in the wine quality data from zip file 

        Args:
            path (str): location of file on computer

        Returns:
            pd.DataFrame: complete wine quality dataset
        """
        zf = ZipFile(path)
        df = pd.read_csv(zf.open('winequality-white.csv'),sep=';')
        self.df = df

    
    def training(self,test_frac: float = 0.1):
        """ Return ONLY the training data.
        Idendify the training data if it does not yet exist
        """
        if self.train is None:
            self._train_test_split(test_frac)    
        return self.train
    
    def testing(self,test_frac: float = 0.1):
        """ Return ONLY the training data.
        Idendify the training data if it does not yet exist
        """
        if self.test is None:
            self._train_test_split(test_frac)
        return self.test

    # if it starts with an underscore you are not allowed
    # to reference it from another file
    def _train_test_split(self, test_frac: float = 0.1):
        all_rows = np.arange(len(self.df))
        np.random.shuffle(all_rows)
        test_n_rows = round(len(self.df)*test_frac)
        test_rows = all_rows[:test_n_rows]
        train_rows = all_rows[test_n_rows:]
        
        self.test = self.df.loc[test_rows].reset_index(drop=True)
        self.train = self.df.loc[train_rows].reset_index(drop=True)

if __name__ == '__main__':
    wq = WineQuality()
    wq.read('data/wine_quality.zip')
    print(wq.df)