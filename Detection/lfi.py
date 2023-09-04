import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from urllib.parse import unquote
import numpy as np
import pickle

class LFIDetector:
    def __init__(self, d2v_model_path, ml_model_paths):
        self.d2v_model_path = d2v_model_path
        self.ml_model_paths = ml_model_paths

    def classify_lfi(self, test_lfi):
        features = self._get_features(test_lfi)
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
            lowerStr = str(line_decode).lower()
            feature1 = int(lowerStr.count('../'))
            feature1 += int(lowerStr.count('./'))
            feature1 += int(lowerStr.count('..//'))
            feature1 += int(lowerStr.count('/etc/'))
            feature1 += int(lowerStr.count('.ini'))
            feature1 += int(lowerStr.count('.log'))
            feature1 += int(lowerStr.count('logs'))
            feature1 += int(lowerStr.count('apache'))
            feature1 += int(lowerStr.count('usr/'))
            feature1 += int(lowerStr.count('/bin/'))
            feature1 += int(lowerStr.count('/opt/'))
            feature1 += int(lowerStr.count('xampp'))
            feature1 += int(lowerStr.count('/root/'))
            feature1 += int(lowerStr.count('/local/'))
            feature1 += int(lowerStr.count('/adm/'))
            feature1 += int(lowerStr.count('/log/'))
            feature1 += int(lowerStr.count('/logs/'))
            feature1 += int(lowerStr.count('/var/'))
            feature1 += int(lowerStr.count('/www/'))
            feature1 += int(lowerStr.count('/proc/'))
            feature1 += int(lowerStr.count('\MySQL'))
            feature1 += int(lowerStr.count('config'))
            feature1 += int(lowerStr.count('ssh'))
            feature1 += int(lowerStr.count('file://'))
            feature1 += int(lowerStr.count('://'))
            feature1 += int(lowerStr.count('/ftp/'))
            feature1 += int(lowerStr.count('/sbin/'))
            feature1 += int(lowerStr.count('httpd/'))
            feature1 += int(lowerStr.count('/cpanel/'))
            feature1 += int(lowerStr.count('.htaccess'))
            feature1 += int(lowerStr.count('etc'))
            # add feature for malicious method/event count
            feature2 = int(lowerStr.count('..\/'))
            feature2 += int(lowerStr.count('boot.ini'))
            feature2 += int(lowerStr.count('win.ini'))
            # add feature for ".js" count
            feature3 = int(lowerStr.count('.ini'))
            feature3 = int(lowerStr.count('.log'))

            # add feature for "javascript" count
            feature4 = int(lowerStr.count('lfi'))
            # add feature for length of the string
            feature5 = int(len(lowerStr))
            # add feature for "<script"  count
            feature6 = int(lowerStr.count('../'))
            feature6 += int(lowerStr.count('%252e%252e'))
            feature6 += int(lowerStr.count('%5c..%5c'))
            feature6 += int(lowerStr.count('..%2f'))
            # add feature for special character count
            feature7 = int(lowerStr.count('&'))
            feature7 += int(lowerStr.count('<'))
            feature7 += int(lowerStr.count('>'))
            feature7 += int(lowerStr.count('"'))
            feature7 += int(lowerStr.count('\''))
            feature7 += int(lowerStr.count('/'))
            feature7 += int(lowerStr.count('%'))
            feature7 += int(lowerStr.count('*'))
            feature7 += int(lowerStr.count(';'))
            feature7 += int(lowerStr.count('+'))
            feature7 += int(lowerStr.count('='))
            feature7 += int(lowerStr.count('%3C'))
            # add feature for http count
            feature8 = int(lowerStr.count('http'))
            
            # append the features
            featureVec = np.append(featureVec,feature1)
            #featureVec = np.append(featureVec,feature2)
            featureVec = np.append(featureVec,feature3)
            featureVec = np.append(featureVec,feature4)
            featureVec = np.append(featureVec,feature5)
            featureVec = np.append(featureVec,feature6)
            featureVec = np.append(featureVec,feature7)
            #featureVec = np.append(featureVec,feature8)
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