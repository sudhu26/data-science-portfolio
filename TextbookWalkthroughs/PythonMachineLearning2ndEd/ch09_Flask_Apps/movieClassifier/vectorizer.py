import sklearn.feature_extraction as feature_extraction
import os
import re
import pickle

curDir = os.path.dirname('__file__')
stop = pickle.load(open(os.path.join(curDir,'ch09_Flask_Apps','movieClassifier','pkl_objects','stopwords.pkl'),'rb'))

def textProcessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = feature_extraction.text.HashingVectorizer(decode_error = 'ignore'
                                                ,n_features = 2**21
                                                ,preprocessor = None
                                                ,tokenizer = textProcessor)