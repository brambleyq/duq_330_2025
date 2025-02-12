import pandas as pd

class Tmdb:
    def __init__(self,path:str):
        """reads in a the movie database data from a csv file

        Args:
            path (str): location of the movie data base csv
        """
        self.df: pd.DataFrame = pd.read_csv(path)
        self._clean_df_columns()
        self._clean_data()
        # going to want to seperate these anyway so its done here
        self.safe = self.df.loc[ ~ self.df['adult']]
        self.adult = self.df.loc[self.df['adult']]
    
    def _clean_df_columns(self):
        """fix column names to be lower case and _ sep"""
        mapper = {
        'id':'id',
        'title':'title',
        'vote_average':'rating',
        'vote_count':'num_votes',
        'status':'status',
        'release_date':'release_date',
        'revenue':'revenue',
        'runtime':'runtime',
        'adult':'adult',
        'budget':'budget',
        'imdb_id':'imdb_id',
        'original_language':'original_language',
        'original_title': 'original_title',
        'overview':'summary',
        'popularity':'popularity',
        'tagline':'tagline',
        'genres':'genres',
        'production_companies':'production_companies',
        'production_countries':'production_countries',
        'spoken_languages':'languages',
        'keywords':'keywords'
        }
        self.df = self.df.rename(columns=mapper)[mapper.values()]

    def _clean_data(self):
        """make the data more usable"""
        self.df.dropna(subset=['title'],inplace=True)
        # to combat repeat titles the year will be put after every title in ()
        self.df['title'] = self.df.apply(lambda row: row['title'] + '' if pd.isna(row['release_date']) else f' ({row['release_date'].split('-')[0]})', axis=1)


if __name__ == "__main__":
    movies = Tmdb("data/TMDB_movie_dataset_v11.csv")
    print(movies.df)
