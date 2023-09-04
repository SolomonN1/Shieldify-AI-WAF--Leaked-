from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from urllib.parse import unquote
import numpy as np
import pickle
import nltk
import concurrent.futures

nltk.download('punkt')

class XSSDetector:
    def __init__(self, d2v_model_path, ml_model_paths):
        self.model = Doc2Vec.load(d2v_model_path)
        self.models = self.load_ml_models(ml_model_paths)
    
    def load_ml_models(self, model_paths):
        models = []
        for path in model_paths:
            with open(path, 'rb') as f:
                loaded_model = pickle.load(f)
            models.append(loaded_model)
        return models
    
    def get_vec(self, text):
        features = []
        for line in text:
            test_data = word_tokenize(line.lower())
            v1 = self.model.infer_vector(test_data)
            line_decode = unquote(line)
            lower_str = line_decode.lower()
            feature1 = sum(lower_str.count(word) for word in ['link', 'object', 'form', 'embed', 'ilayer', 'layer', 'style', 'applet', 'meta', 'img', 'iframe', 'marquee', 'script'])
            feature2 = sum(lower_str.count(word) for word in ['exec', 'fromcharcode', 'eval', 'alert', 'getelementsbytagname', 'write', 'unescape', 'escape', 'prompt', 'onload', 'onclick', 'onerror', 'onpage', 'confirm'])
            feature3 = lower_str.count('.js')
            feature4 = lower_str.count('javascript')
            feature5 = len(lower_str)
            feature6 = sum(lower_str.count(word) for word in ['script', '<script', '&lt;script', '%3cscript', '%3c%73%63%72%69%70%74'])
            feature7 = sum(lower_str.count(symbol) for symbol in ['&', '<', '>', '"', '\'', '/', '%', '*', ';', '+', '=', '%3C'])
            feature8 = lower_str.count('http')
            feature_vec = np.append(v1, [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8])
            features.append(feature_vec)
        
        return features
    
    def classify_xss(self, test_xss):
        X_new = self.get_vec(test_xss)
        X_new = np.array(X_new)

        def classify_sample(i):
            y_pred = [model.predict(X_new[i].reshape(1, -1)) for model in self.models]
            score = sum(0.175 * y_pred_j[0] for y_pred_j in y_pred)  # Update the weights and adjust accordingly for each model

            if score >= 0.5:
                return True
            else:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            results = list(executor.map(classify_sample, range(len(X_new))))

        return results