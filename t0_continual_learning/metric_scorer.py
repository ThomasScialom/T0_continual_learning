import os
import json
import numpy as np
from typing import List
import scipy.stats
import syllables
from datasets import load_metric

import matplotlib.pyplot as plt

from t0_continual_learning.config_variables import evaluation_new_tasks, evaluation_T0evalsets

class MetricScorer():
  def __init__(self, path_dscores=None):
    
    self.d_scores = {}
    self.path_dscores = path_dscores

    if path_dscores:
      with open(path_dscores, 'r') as f:
        self.d_scores = json.load(f)
  
  def getScore(self, prompt_config, path_ex, path_pred):
    
    format = lambda xs: xs# [x.strip() for x in xs]

    preds = format(self.openJsonFile(path_pred)['hyps'])

    exs = self.openJsonFile(path_ex)
    self.preds = preds
    refs = format(exs['tgt'])

    list_refs = refs
    if isinstance(refs[0], str):
      list_refs = [[r] for r in refs]

    assert len(preds) == len(refs)
      
    d_res = {}
    for metric in prompt_config['metrics']:

      if metric == "rouge":
        d_res = {**d_res, **self.computeRouge(preds, refs)}

      if metric == "bleu":
        d_res = {**d_res, **self.computeBleu(preds, list_refs)}
      
      if metric == "bertscore":
        d_res = {**d_res, **self.computeBERTScore(preds, list_refs)}
        
      if metric == "sari":
        d_res = {**d_res, **self.computeSari(preds, list_refs, exs['src'])} 
            
      if metric == "accuracy":
        d_res = {**d_res, **self.computeAcc(preds, refs)}
      
      if "constrain" in metric:
        d_res = {**d_res, **self.computeConstrain(preds, refs, exs['src_info'], metric)}

      if metric == "haikuMetric":
        bleu_score = self.computeBleu(preds, list_refs)['bleu']
        d_res = {**d_res, **self.computeHaiku(preds, refs, exs['src'], bleu_score)} 
      
      if metric == "firstWordSim":
        d_res = {**d_res, **self.computeFirstWordSim(preds, refs)}

    return d_res
        
  @staticmethod
  def openJsonFile(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data
  
  def computeBERTScore(self, preds, list_refs):
    
    metric = load_metric("bertscore", model_type='microsoft/deberta-large-mnli')
    metric.add_batch(predictions=preds, references=list_refs)
    scores = metric.compute(lang='en')

    return {'BERTScore(f1)': np.average(scores['f1'])}
  
  def computeRouge(self, preds, refs):
    
    rouge = load_metric("rouge")
    rouge.add_batch(predictions=preds, references=refs)
    d_res = rouge.compute()

    return {k:v.mid.fmeasure  for k, v in d_res.items()}

  def computeSari(self, preds, list_refs, srcs):
    
    sari = load_metric("sari")
    sari.add_batch(predictions=preds, references=list_refs)
    d_res = sari.compute(sources=srcs)

    return d_res

  def computeBleu(self, preds, list_refs):
    
    bleu = load_metric("bleu")
    bleu.add_batch(
        predictions=[pred.split() for pred in preds], 
        references=[[ref.split() for ref in refs] for refs in list_refs]
    )
    d_res = bleu.compute()

    return {'bleu': d_res['bleu']}

  def computeAcc(self, preds, refs):
    
    total_correct = sum([pred==ref for pred, ref, in zip(preds, refs)])
    total_nb = len(preds)

    return {"accuracy": total_correct/total_nb}

  def computeConstrain(self, preds, refs, src_infos, metric):

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


  def computeHaiku(self, preds, refs, srcs, bleu_score):
    
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

  def computeFirstWordSim(self, preds, refs):    

    def jensen_shannon_distance(p, q):
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


    def getFirstTok(sent):
      tok = ""
      if sent:
        tok = sent.split()[0].lower()

      return tok

    def getTok2idx(all_sents):
      tok2idx = {}

      count = 0
      for sent in all_sents:
        
        tok = getFirstTok(sent)
        if tok not in tok2idx:
          tok2idx[tok] = count
          count += 1

      return tok2idx


    def getArray(tok2idx, sents):

      arr = [0] * len(tok2idx)

      for sent in sents:
        tok = getFirstTok(sent)
        arr[tok2idx[tok]] += 1

      return arr

    tok2idx = getTok2idx(preds + refs)
    d = jensen_shannon_distance(getArray(tok2idx, preds), getArray(tok2idx, refs))
    return {'jensenFirstToken': 1/d}

  def getAllScores(self, path_folder_preds, path_folder_data, init_from_sratch=False, evaluation_config=None):
    
    if init_from_sratch == True:
      self.d_scores = {}

    if not evaluation_config:
      evaluation_config = {**evaluation_new_tasks, **evaluation_T0evalsets}

    list_files = os.listdir(path_folder_preds)
    nb_files = len(list_files)
    for i, file in enumerate(list_files):
      
      # if folder
      if '.' not in file:
        continue
      
      if i % 50 == 0:
        print(f'{i}/{nb_files}')

      if "T0_3B" in file or "T0pp" in file:
        evalset, eval_mode , prompt_name, model_name, _ = file.split('.')
        assert model_name == 'T0_3B' or model_name == 'T0pp'
        key = f'{model_name}.{evalset}.{eval_mode}.{prompt_name}'
      elif "sequencial" in file:
        evalset, eval_mode , prompt_name, model_size, rehearsal, _, model_name, _, model_from, step, _ = file.split('.')
        key = f'{model_name}.{model_size}.{rehearsal}.{step}.{evalset}.{eval_mode}.{prompt_name}.{model_from}'
      else:
        evalset, eval_mode , prompt_name, model_size, rehearsal, model_name, step, _ = file.split('.')
        key = f'{model_name}.{model_size}.{rehearsal}.{step}.{evalset}.{eval_mode}.{prompt_name}'

      prompt_config = evaluation_config[evalset][eval_mode][prompt_name]

      if key in self.d_scores:
        if self.areMetricsDone(prompt_config['metrics'], self.d_scores[key]) == True:
          continue 

      path_pred = os.path.join(path_folder_preds, file)
      path_ex = os.path.join(path_folder_data, evalset, f'{prompt_name}.{eval_mode}.json')
      
      self.d_scores[key] = self.getScore(prompt_config, path_ex, path_pred)

    if self.path_dscores:
      with open(self.path_dscores, 'w') as f:
        json.dump(self.d_scores, f, indent=3)
        
  def areMetricsDone(self, metrics, dict_res):
    all_metric_done = True
        
    for metric in metrics:
      if metric == 'rouge' and 'rouge1' not in dict_res:
        all_metric_done = False
      if metric == 'bleu' and 'bleu' not in dict_res:
        all_metric_done = False
      if metric == 'bertscore' and 'BERTScore(f1)' not in dict_res:
        all_metric_done = False
      if metric == 'constrain_contain' and 'contain' not in dict_res:
        all_metric_done = False
      if metric == 'constrain_end' and 'end' not in dict_res:
        all_metric_done = False
      if metric == 'constrain_start' and 'start' not in dict_res:
        all_metric_done = False
      if metric == 'accuracy' and 'accuracy' not in dict_res:
        all_metric_done = False
      if metric == 'sari' and 'sari' not in dict_res:
        all_metric_done = False
      if metric == 'haikuMetric' and 'comma' not in dict_res:
        all_metric_done = False
      if metric == 'firstWordSim' and 'jensenFirstToken' not in dict_res:
        all_metric_done = False

    return all_metric_done
