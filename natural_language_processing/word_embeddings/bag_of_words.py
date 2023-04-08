import pandas as pd
from typing import Iterable
from tqdm import tqdm
import warnings 

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class BagOfWords:
    def __init__(self, count: bool = False) -> None:
        """
        Code Example:
            model = BagOfWords(count=True)
            result = model.fit_transform(documents)

        count:
            if count is True, then the value of the cell will be the count of the word in the document
            else the value of the cell will be 1 if the word is present in the document else 0
        """
        self.documents = None
        self.no_of_documents = None
        self.bow = None
        self.count = count

    def fit_transform(self, documents: Iterable[Iterable]) -> pd.DataFrame:
        """
        documents: Contains list of tokenized documents
        Example : 
            documents = [
                ["hritik", "is", "awesome", "."],
                ["He", "lives", "in", "Manali", "."]
            ]
        
        Returns a dataframe of bag of words
        """
        return self._fit_transform(documents=documents)

    def _fit_transform(self, documents) -> pd.DataFrame:
        """
        documents: Contains list of tokenized documents
        Example : 
            documents = [
                ["hritik", "is", "awesome", "."],
                ["He", "lives", "in", "Manali", "."]
            ]
        Returns a dataframe of bag of words
        """
        if documents is None:
            raise Exception("Please provide documents")
        self.documents = documents
        self.no_of_documents = len(self.documents)
        self._zero_column = pd.Series(0,index = range(self.no_of_documents))
        self.bow = pd.DataFrame(index = range(self.no_of_documents))
        if self.count:
            for document_id, document in enumerate(tqdm(self.documents)):
                for token in document:
                    token = token.lower()
                    if token not in self.bow.columns:
                        self.bow[token] = self._zero_column
                    #self.bow = pd.concat([self.bow,self._zero_column.rename(token)], axis = 1)
                    self.bow.loc[document_id,token] +=1
            return self.bow
        else:
            for document_id, document in enumerate(tqdm(self.documents)):
                for token in document:
                    token = token.lower()
                    if token not in self.bow.columns:
                        self.bow[token] = self._zero_column
                    #self.bow = pd.concat([self.bow,self._zero_column.rename(token)], axis = 1)
                    self.bow.loc[document_id,token] = 1
        
        return self.bow