from sklearn.base import BaseEstimator, TransformerMixin
import re

class PrepProcesor(BaseEstimator, TransformerMixin): 
    def transform(self, X, y=None):
            data['label'] = data['label'].astype(int)
            data['label'] = data['label'].astype('category')
            data['text'] = data['text'].astype(str)
            data['text_without_stop'] = data['text_without_stop'].astype(str)
            return data 

columns = ['text', 'label', 'text_without_stop', 'fonte']