from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import pickle
import os
import json


class ReusableClassifier:
    def __init__(self, model_type: str):
        """Create a classifier, storing a model and metadata.

        Args:
            model_type (str): cal include random forest, logistic
                regression, etc.
        """
        # Initilizing the scalar variable fo rhuse in the future
        # Add all shared bariables ot the init function
        self._scalar = None
        self._metadata = None

        self._model_type = model_type
        if model_type == 'logistical_regression':
            self._model = self._create_logistic_progression()
        elif model_type == 'random_forest':
            self._model = self._create_random_forest()
        elif model_type == 'xgboost':
            self._model = self._create_xgboost()
        else:
            raise ValueError('not a valid model type')
    
    def train(self,features:pd.DataFrame,
              labels:pd.Series,
              test_frac: float = 0.1):
        """Train the model from pandas data

        Args:
            features (pd.DataFrame): features, dataframe
            labels (pd.Series): input labels
            test_fraction (float, optional): fraction of datato preserve
                for testing. Defaults to 0.1.
        """
        # X is another term for featurs, y is another term for labels

        # we need to scale the data
        # 1. we can set he min to 0 and the max to 1
        # BUT, we need ot consider outliers
        # we could remove outliers

        # or 2. we can use standardardization by standard normal
        # so for each data point centerit around the mean then divid it by the standard deviation
        self._scalar = StandardScaler()
        self._scalar.fit(features)
        features = self._scalar.transform(features)

        features_train, features_test, labels_train, lables_test = train_test_split(features, labels, test_size=test_frac)

        self._model.fit(features_train, labels_train)
        pred_labels = self._model.predict(features_test)

        # Manual accuracy
        accuracy = (pred_labels == lables_test).mean()

        self._metadata = {}
        self._metadata['training_rows'] = len(features_train)
        self._metadata['accuracy'] = accuracy
        self._metadata['model_type'] = self._model_type

        print(f'accuracy for the test set was {accuracy*100}%')


    def predict(self,features:pd.DataFrame):
        features = self._scalar.transform(features)
        return self._model.predict(features)
    
    def predict_proba(self,features:pd.DataFrame):
        features = self._scalar.transform(features)
        return [ps[1] for ps in self._model.predict_proba(features)]

    def save(self,path:str):
        """save model

        Args:
            path (str): _description_
        """
        model_path, ext = os.path.splitext(path)
        scalar_path = model_path + '_scalar.pkl'
        metadata_path = model_path + '.json'
        model_path = model_path + '.pkl'

        with open(model_path, 'wb') as fp:
            pickle.dump(self._model, fp)
        with open(scalar_path, 'wb') as fp:
            pickle.dump(self._scalar, fp)
        with open(metadata_path, 'w') as fp:
            json.dump(self._metadata, fp)
            

    def load(self,path:str):
        model_path, ext = os.path.splitext(path)
        scalar_path = model_path + '_scalar.pkl'
        metadata_path = model_path + '.json'
        model_path = model_path + '.pkl'

        with open(model_path, 'rb') as fp:
            self._model = pickle.load(fp)
        with open(scalar_path, 'rb') as fp:
            self._scalar = pickle.load(fp)
        with open(metadata_path, 'r') as fp:
            self._metadata = json.load(fp)

    def _asses_tf_fraction(self, labels:pd.Series):
        """thows an error for dramatically unweighted data

        Args:
            labels (pd.Series): input labels
        """
        total_trues = labels.sum()
        if  total_trues> 0.8*len(labels):
            raise ValueError('Too many trues')
        elif total_trues < 0.2*len(labels):
            raise ValueError('Too many falses')

    def _create_logistic_progression(self):
        """create a new logistic regression model from sklearn
        """
        return LogisticRegression()
    
    def _create_random_forest(self):
        """create a new logistic regression model from sklearn"""
        return RandomForestClassifier()
    
    def _create_xgboost(self):
        """create a new xgboost model"""
        return xgb.XGBClassifier()
    
    
if __name__ == "__main__":
    import winequality
    wq = winequality.WineQuality()
    wq.read('data/wine+quality.zip')
    df = wq.df

    labels = df['quality'] > 5
    features = [['acitity','sulphates','alcohol']]
    features = df.drop(['quality'],axis=1)

    log_reg = ReusableClassifier('logistical_regression')
    log_reg.train(features,labels)
    log_reg.save('data/model_log_reg')
    log_reg.load('data/model_log_reg')

    ran_for = ReusableClassifier('random_forest')
    ran_for.train(features,labels)
    ran_for.save('data/model_ran_for')
    ran_for.load('data/model_ran_for')