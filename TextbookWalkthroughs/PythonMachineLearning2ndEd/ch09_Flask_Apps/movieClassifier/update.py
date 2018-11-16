import pickle
import sqlite3
import numpy as np
import os

from vectorizer import vect

def update_model(db_path, model, batch_size = 10000):
    conn = sqlite3.connect)db_path
    c = conn.cursor()
    c.execute(" SELECT * FROM db_review ")
    result = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:,0]
        y = data[:,1].astype(int)

        classes = np.array([0,1])
        XTrain = vect.transform(X)
        model.partial_fit(XTrain, y, classes = classes)
        results = c.fetchmany(batch_size)
    
    conn.close()
    return model

cur_dir = os.path.dirname('__file__')

cur_dir = os.path.dirname('__file__')
clf = pickle.load(open(os.path.join(cur_dir
                                    ,'ch09_Flask_Apps'
                                    ,'movieClassifier'
                                    ,'pkl_objects'
                                    ,'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

clf = update_model(db_path = db, model = clf, batch_size = 10000)

# This section below updates the classifier.pkl file permanently
# pickle.dump(clf, open(os.path.join(cur_dir
#                                     ,'ch09_Flask_Apps'
#                                     ,'movieClassifier'
#                                     ,'pkl_objects'
#                                     ,'classifier.pkl'), 'wb')
#                                 ,protocol = 4)