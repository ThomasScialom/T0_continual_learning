import os
import json
import time

from t0_continual_learning import custom_metrics
from t0_continual_learning.config_variables import evaluation_new_tasks, evaluation_T0evalsets

class MetricScorer():
  def __init__(self, path_dscores=None):
    
    self.d_scores = {}
    self.path_dscores = path_dscores

    if path_dscores:
      with open(path_dscores, 'r') as f:
        self.d_scores = json.load(f)
    
    self.d_metric_map_names = {
        'rouge': 'rouge1',
        'bleu': 'bleu',
        'bertscore': 'BERTScore(f1)',
        'constrain_contain': 'contain',
        'constrain_end': 'end',
        'constrain_start': 'start',
        'accuracy': 'accuracy',
        'sari': 'sari',
        'haikuMetric': 'comma',
        'firstWordSim': 'jensenFirstToken',
        'CLF_acc': 'CLF_acc',
        'selfbleu': 'selfbleu',
    }
    
  def getScore(self, prompt_config, path_ex, path_pred, path_folder_data, evalset, prompt_name, d_res = {}):
    
    format = lambda xs: xs# [x.strip() for x in xs]

    preds = format(self.openJsonFile(path_pred)['hyps'])

    exs = self.openJsonFile(path_ex)
    self.preds = preds
    refs = format(exs['tgt'])

    list_refs = refs
    if isinstance(refs[0], str):
      list_refs = [[r] for r in refs]

    assert len(preds) == len(refs)
    
    for metric in prompt_config['metrics']:
      
      if self.d_metric_map_names[metric] in d_res:
        continue
      
      if metric == "rouge":
        d_res = {**d_res, **custom_metrics.computeRouge(preds, refs)}

      elif metric == "bleu":
        d_res = {**d_res, **custom_metrics.computeBleu(preds, list_refs)}
        
      elif metric == "selfbleu":
        d_res = {**d_res, **custom_metrics.computeSelfBleu(preds)}

      elif metric == "bertscore":
        d_res = {**d_res, **custom_metrics.computeBERTScore(preds, list_refs)}
        
      elif metric == "sari":
        d_res = {**d_res, **custom_metrics.computeSari(preds, list_refs, exs['src'])} 
            
      elif metric == "accuracy":
        d_res = {**d_res, **custom_metrics.computeAcc(preds, refs)}
      
      elif "constrain" in metric:
        d_res = {**d_res, **custom_metrics.computeConstrain(preds, refs, exs['src_info'], metric)}

      elif metric == "haikuMetric":
        bleu_score = custom_metrics.computeBleu(preds, list_refs)['bleu']
        d_res = {**d_res, **custom_metrics.computeHaiku(preds, refs, exs['src'], bleu_score)} 
      
      elif metric == "firstWordSim":
        firstWordSim = custom_metrics.FirstWordSim()
        d_res = {**d_res, **firstWordSim.compute(preds, refs)}
        
      elif metric == "CLF_acc":
        
        if evalset == 'twitter_top20':
          label_name = 'author'
        elif evalset == 'empathetic_dialogues':
          label_name = 'context_emotion'
        clf = custom_metrics.Clf(path_folder_data, evalset, prompt_name, label_name)
        d_res = {**d_res, **clf.compute_score(preds)}

    return d_res
        
  @staticmethod
  def openJsonFile(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data

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
      
      if evalset not in evaluation_config:
        print(f'evalset not in evaluation_config, continue: {evalset}')
        continue
        
      prompt_config = evaluation_config[evalset][eval_mode][prompt_name]

      d_key = {}
      if key in self.d_scores:
        d_key = self.d_scores[key]
        if self.areMetricsDone(prompt_config['metrics'], self.d_scores[key]) == True:
          continue 

      path_pred = os.path.join(path_folder_preds, file)
      path_ex = os.path.join(path_folder_data, evalset, f'{prompt_name}.{eval_mode}.json')
      
      print(f'{i}/{nb_files}...... {path_pred}')
      start_time = time.time()
      self.d_scores[key] = self.getScore(prompt_config, path_ex, path_pred, path_folder_data, evalset, prompt_name, d_res=d_key)
      print(f'\t..... took {round(time.time() - start_time, 2)}')
      
    if self.path_dscores:
      with open(self.path_dscores, 'w') as f:
        json.dump(self.d_scores, f, indent=3)
        
  @staticmethod
  def areMetricsDone(metrics, dict_res):
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
      if metric == 'CLF_acc' and 'CLF_acc' not in dict_res:
        all_metric_done = False
      if metric == 'selfbleu' and 'selfbleu' not in dict_res:
        all_metric_done = False

    return all_metric_done
