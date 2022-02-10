import os, json
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
import _pickle as cPickle

class Clf():
  
  """
  Usage:
  1) load the clf for a task:
  path_folder_data = f'{GLOBAL_PATH}/data'
  evalset = 'twitter_top20'
  prompt_name = 'tweet_as+about'
  label_name = 'author'
  clf = Clf(path_folder_data, evalset, prompt_name, label_name)
  
  2) infer:
  # path_prediction the path containing the predictions of your model
  print(clf.compute_score(path_prediction))
  """


  def __init__(self, path_folder_data, evalset, prompt_name, label_name):
    self.path_folder_data = path_folder_data
    self.evalset = evalset
    self.prompt_name = prompt_name
    self.label_name = label_name

    self.key_name = f'{evalset}.{prompt_name}.{label_name}'

    path_model = f'{self.key_name}.model.pkl'
    path_count_vectorizer = f'{self.key_name}.count_vectorizer.pkl'

    if os.path.exists(path_model):
      # load it
      with open(path_model, 'rb') as fid:
          self.model = cPickle.load(fid)
      with open(path_count_vectorizer, 'rb') as fid:
          self.count_vectorizer = cPickle.load(fid)
    else:
      self.model = RidgeClassifier() #GaussianNB()
      self.count_vectorizer = CountVectorizer(binary=True)
      self.train_model()
      # save the classifier
      with open(path_model, 'wb') as fid:
        cPickle.dump(self.model, fid)  
      with open(path_count_vectorizer, 'wb') as fid:
        cPickle.dump(self.count_vectorizer, fid)  

    #transform test data
    X_test, y_test = self.get_data('test')
    self.y_test = y_test
    predictions = self.get_preds(X_test)
    print("Accuracy clf:", self.accuracy_score(y_test, predictions))    

  def get_data(self, eval_mode):

    path_ex = os.path.join(self.path_folder_data, self.evalset, f'{self.prompt_name}.{eval_mode}.json')

    with open(path_ex, 'r') as f:
      data = json.load(f)

    nb_ex = len(data['src_info'])
    outputs = [data['tgt'][idx] for idx in range(nb_ex)]
    labels = [data['src_info'][idx][self.label_name] for idx in range(nb_ex)]
    
    assert len(outputs) == len(labels)

    return outputs, labels

  def train_model(self):
    
    #fit training data
    X_train, y_train = self.get_data('train')
    training_data = self.count_vectorizer.fit_transform(X_train).toarray()
    self.model.fit(training_data, y_train)

  @staticmethod
  def accuracy_score(y_true, y_pred):
    return np.average([y1 == y2 for y1, y2 in zip(y_true, y_pred)])

  def get_preds(self, X_test):
    testing_data = self.count_vectorizer.transform(X_test).toarray()
    predictions = self.model.predict(testing_data)

    return predictions

  def compute_score(self, path_pred):
    with open(path_pred, 'r') as f:
      outputs = json.load(f)['hyps']

    predictions = self.get_preds(outputs)
    return {'CLF_acc': self.accuracy_score(self.y_test, predictions)}


