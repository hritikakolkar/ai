import pandas as pd
from numpy import log
from bag_of_words import BagOfWords
from typing import Iterable

class TfIdf(BagOfWords):
    def __init__(self) -> None:
        """
        Code Example:
            model = TfIdf()
            result = model.fit_transform(documents)
        """
        super().__init__(count = True)
        
    def fit_transform(self, documents: Iterable[Iterable]) -> pd.DataFrame:
        """
        documents: Contains list of tokenized documents
        Example : 
            documents = [
                ["hritik", "is", "awesome", "."],
                ["He", "lives", "in", "Manali", "."]
            ]
        Returns a dataframe of tern frequency inverse document frequency
        """
        return self._fit_transform(documents)
    
    def _fit_transform(self, documents: Iterable[Iterable]) -> pd.DataFrame:
        """
        documents: Contains list of tokenized documents
        Example : 
            documents = [
                ["hritik", "is", "awesome", "."],
                ["He", "lives", "in", "Manali", "."]
            ]
        Returns a dataframe of tern frequency inverse document frequency
        """
        bow = super()._fit_transform(documents)
        tf = bow.div(bow.sum(axis = 1), axis = 0)
        token_appearances = (bow!=0).sum(axis = 0)
        idf = log(self.no_of_documents / token_appearances)
        tf_idf = tf.mul(idf, axis = 1)
        return tf_idf
