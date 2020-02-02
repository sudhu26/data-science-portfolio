from sklearn.feature_extraction.text import CounterVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
import os
import re
import pickle

cur_dir = os.path.dirname('__file__')
stop = pickle.load(open(os.path.join(cur_dir,'ch09_Flask_Apps','movieClassifier','pkl_objects','stopwords.pkl'),'rb'))

def text_processor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error = 'ignore'
                                                ,n_features = 2**21
                                                ,preprocessor = None
                                                ,tokenizer = text_processor)