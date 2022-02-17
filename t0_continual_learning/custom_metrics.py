import os, json
import numpy as np
from typing import List

from datasets import load_metric
from bert_score import score as bscore

import scipy.stats
import syllables

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
  print(clf.compute_score(evaluated_predictions))
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

  def compute_score(self, outputs):

    clf_predictions = self.get_preds(outputs)
    return {'CLF_acc': self.accuracy_score(self.y_test, clf_predictions)}

def computeBERTScore(preds, list_refs):
  
  P, R, F1 = bscore(preds, list_refs, lang="en", model_type='microsoft/deberta-large-mnli')
  return {'BERTScore(f1)': F1.mean().item()}

def computeRouge(preds, refs):

  rouge = load_metric("rouge")
  rouge.add_batch(predictions=preds, references=refs)
  d_res = rouge.compute()

  return {k:v.mid.fmeasure  for k, v in d_res.items()}

def computeSari(preds, list_refs, srcs):

  sari = load_metric("sari")
  sari.add_batch(predictions=preds, references=list_refs)
  d_res = sari.compute(sources=srcs)

  return d_res

def computeBleu(preds, list_refs):

  bleu = load_metric("bleu")
  bleu.add_batch(
      predictions=[pred.split() for pred in preds], 
      references=[[ref.split() for ref in refs] for refs in list_refs]
  )
  d_res = bleu.compute()

  return {'bleu': d_res['bleu']}

def computeSelfBleu(preds):

  bleu = load_metric("bleu")
  
  sum_bleu = 0
  for i, pred in enumerate(preds):
    
    refs = preds[:i] + preds[i+1:]
    bleu.add_batch(
        predictions=[pred.split()], 
        references=[[ref.split() for ref in refs]]
    )
    sum_bleu += bleu.compute()['bleu']

  return {'selfbleu': sum_bleu/len(preds)}

def computeAcc(preds, refs):

  total_correct = sum([pred==ref for pred, ref, in zip(preds, refs)])
  total_nb = len(preds)

  return {"accuracy": total_correct/total_nb}


def computeConstrain(preds, refs, src_infos, metric):

  correct = 0
  for i, (src_info, pred) in enumerate(zip(src_infos, preds)):
    constr_type = src_info["constrain_type"]
    assert metric == f'constrain_{constr_type}'

    span_to_insert = src_info["TO_REPLACE_1"] 

    if constr_type == 'start':
      if span_to_insert == pred[:len(span_to_insert)]:
        correct += 1

    if constr_type == 'contain':
      if span_to_insert in pred: 
        correct += 1

    if constr_type == 'end':
      if span_to_insert == pred[-len(span_to_insert):]:
        correct += 1

  return {constr_type: correct/len(src_infos)}

  
def computeHaiku(preds, refs, srcs, bleu_score):

  normaliseDifScore = lambda nb_tgt, nb_hyp: 1-abs(nb_tgt - nb_hyp)/max([nb_tgt, nb_hyp])
  constrainScorer = lambda src, hyp: 1 if ' '.join(src.split("'")[1:]).strip() in hyp else 0

  d_score = {
      'syllable': 0,
      'comma': 0,
      'constrain': 0,
      'bleu': bleu_score
  }

  for tgt, hyp, src in zip(refs, preds, srcs):
    d_score['syllable'] += normaliseDifScore(syllables.estimate(tgt), syllables.estimate(hyp)) 
    d_score['comma'] += normaliseDifScore(len(tgt.split(',')), len(hyp.split(','))) 
    d_score['constrain'] += constrainScorer(src, hyp) 

  for k in ['syllable', 'comma', 'constrain']:
    d_score[k] /= len(preds)

  d_score['eq_weighted'] = sum(d_score.values()) / len(d_score)

  return d_score

class FirstWordSim():    
  
  def __init__(self):
    pass
  
  def compute(self, preds, refs):
    tok2idx = self.getTok2idx(preds + refs)
    d = self.jensen_shannon_distance(self.getArray(tok2idx, preds), self.getArray(tok2idx, refs))
    return {'jensenFirstToken': 1/d}
  
  def jensen_shannon_distance(self, p, q):
      """
      Thanks to @sourcedexter (https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d)
      method to compute the Jenson-Shannon Distance 
      between two probability distributions
      """

      # convert the vectors into numpy arrays in case that they aren't
      p = np.array(p)
      q = np.array(q)

      # calculate m
      m = (p + q) / 2

      # compute Jensen Shannon Divergence
      divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

      # compute the Jensen Shannon Distance
      distance = np.sqrt(divergence)

      return distance


  def getFirstTok(self, sent):
    tok = ""
    if sent:
      tok = sent.split()[0].lower()

    return tok

  def getTok2idx(self, all_sents):
    
    tok2idx = {}
    count = 0
    for sent in all_sents:

      tok = self.getFirstTok(sent)
      if tok not in tok2idx:
        tok2idx[tok] = count
        count += 1

    return tok2idx

  def getArray(self, tok2idx, sents):

    arr = [0] * len(tok2idx)

    for sent in sents:
      tok = self.getFirstTok(sent)
      arr[tok2idx[tok]] += 1

    return arr

  

