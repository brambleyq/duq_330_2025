import pandas as pd
from zipfile import ZipFile


def read(path: str) -> pd.DataFrame:
    """Read in NPI data and rename columns"""

    df = pd.read_csv(path)
    mapper = {
            'NPI': 'npi',
            'Healthcare Provider Taxonomy Code_1': 'taxonomy_code',
            'Provider Last Name (Legal Name)': 'last_name',
            'Provider First Name': 'forename',
            'Provider First Line Business Practice Location Address': 'address',
            'Certification Date': 'cert_date',
            'Provider Business Practice Location Address State Name': 'city',
            'Provider Business Practice Location Address State Name': 'state',
            'Provider Business Practice Location Address Country Code (If outside U.S.)': 'country'
        }
    
    df = df.rename(columns=mapper)[mapper.values()]
    df.dropna(subset=['forename','last_name'],inplace=True)
    df['name'] = df.apply(lambda row: row['forename'] +' '+ row['last_name'],axis=1)
    return df

    #There are 850 missing values for taxonomy_code, address, state, and country
    #There are 6,259 missing last names and 6,262 missing first names
    #There are 1,505 missing values for cert_date


if __name__ == '__main__':
    df = read("data/npidata_pfile_20050523-20250309.csv")
    print(df.head())