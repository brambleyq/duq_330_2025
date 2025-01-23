import pandas as pd
from zipfile import ZipFile

def read_wq(path: str) -> pd.DataFrame:
    """read in the wine quality data from zip file 

    Args:
        path (str): location of file on computer

    Returns:
        pd.DataFrame: complete wine quality dataset
    """
    zf = ZipFile(path)
    df = pd.read_csv(zf.open('winequality-white.csv'),sep=';')
    return df

if __name__ == '__main__':
    df = read_wq('data/wine+quality.zip')

    print(df)