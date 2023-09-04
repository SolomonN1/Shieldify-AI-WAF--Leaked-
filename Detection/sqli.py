import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from urllib.parse import unquote
import numpy as np
import pickle

class SQLiDetector:
    def __init__(self, d2v_model_path, ml_model_paths):
        self.d2v_model_path = d2v_model_path
        self.ml_model_paths = ml_model_paths

    def classify_sqli(self, test_sqli):
        features = self._get_features(test_sqli)
        predictions = self._make_predictions(features)
        return predictions

    def _get_features(self, text):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text)]
        vec_size = 20
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
        model.build_vocab(tagged_data)
        features = []

        for i, line in enumerate(text):
            featureVec = [model.dv[i]]
            line_decode = unquote(line)
            line_decode = line_decode.replace(" ", "")
            lower_str = str(line_decode).lower()
            feature1 = lower_str.count('select') + lower_str.count('from') + lower_str.count('where') + \
                lower_str.count('like') + lower_str.count('join') + lower_str.count('inner') + \
                lower_str.count('order') + lower_str.count('insert') + lower_str.count('update') + \
                lower_str.count('delete') + lower_str.count('create') + lower_str.count('drop') + \
                lower_str.count('union') + lower_str.count('distinct') + lower_str.count('count') + \
                lower_str.count('not') + lower_str.count('and') + lower_str.count('or') + \
                lower_str.count('case') + lower_str.count('when') + lower_str.count('then') + lower_str.count('null') + \
                lower_str.count('wait') + lower_str.count('delay') + lower_str.count('NULL') + lower_str.count('ALL') + \
                lower_str.count('"') + lower_str.count("'") + lower_str.count('_') + lower_str.count('-') + \
                lower_str.count('=') + lower_str.count(')') + lower_str.count('*') + lower_str.count(')') + \
                lower_str.count('^') + lower_str.count('&') + lower_str.count('#') 
                
            feature3 = lower_str.count('.sql') + lower_str.count('.mdb') + \
                lower_str.count('.accdb') + lower_str.count('.db') + lower_str.count('.sqlite') + \
                lower_str.count('.db2') + lower_str.count('.ora') + lower_str.count('.ora)') + \
                lower_str.count('.ora(') + lower_str.count('.mysql') + lower_str.count('.postgresql') + \
                lower_str.count('.msaccess') + lower_str.count('.pg') + lower_str.count('.mssql') + \
                lower_str.count('.pgsql') + lower_str.count('.mysqli') + lower_str.count('.sqlite3') + \
                lower_str.count('.sqlite2') + lower_str.count('.pgdump') + lower_str.count('.dump')
                
            feature5 = len(lower_str)
                
            feature6 = lower_str.count('SELECT') + lower_str.count('UNION') + \
                lower_str.count('INSERT') + lower_str.count('UPDATE') + lower_str.count('DELETE') + \
                lower_str.count('FROM') + lower_str.count('WHERE') + lower_str.count('GROUP BY') + \
                lower_str.count('ORDER BY') + lower_str.count('HAVING') + lower_str.count('AND') + \
                lower_str.count('OR') + lower_str.count('LIKE') + lower_str.count('LIMIT')

            feature7 = lower_str.count('-') + lower_str.count('+') + lower_str.count('=') + lower_str.count('%3C') + \
                        lower_str.count('=') + lower_str.count('/') + lower_str.count('%') + lower_str.count('*') + \
                        lower_str.count(';') 
            feature8 = lower_str.count('http')
            
            # append the features
            featureVec = np.append(featureVec,feature1)
            #featureVec = np.append(featureVec,feature2)
            featureVec = np.append(featureVec,feature3)
            featureVec = np.append(featureVec,feature5)
            featureVec = np.append(featureVec,feature6)
            featureVec = np.append(featureVec,feature7)
            featureVec = np.append(featureVec,feature8)
            #print(featureVec)
            features.append(featureVec)
        return features


    def _make_predictions(self, features):
        predictions = []
        
        for i in range(len(features)):
            xnew = [features[i]]
            ynew = []

            for model_path in self.ml_model_paths:
                loaded_model = pickle.load(open(model_path, 'rb'))
                ynew.append(loaded_model.predict(xnew))

            score = np.dot([.175, .15, .05, .075, .25, .3], ynew)

            if score >= 0.5:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions