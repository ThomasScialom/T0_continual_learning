import json
import os
import matplotlib.pyplot as plt
from datasets import load_metric

from t0_continual_learning.config_variables import evaluation_new_tasks, evaluation_T0evalsets

list_zero_shot = [
    ('rte', "can we infer", 'validation'), 
    ('copa', "choose", 'validation'), 
    ('wic', "same_sense", 'validation'), 
    ('winogrande', "fill in the blank", 'validation'),
    ('hellaswag', "__RANDOM__", 'validation'),
    ('anli', "__RANDOM__", 'test_r1'),
    ('cb', "__RANDOM__", 'validation'),
    ('wsc', "__RANDOM__", 'validation'),
    ('story_cloze', "__RANDOM__", 'validation'),
  ]

class MetricScorer():
  def __init__(self, path_dscores=None):
    
    self.d_scores = {}
    self.path_dscores = path_dscores

    if path_dscores:
      with open(path_dscores, 'r') as f:
        self.d_scores = json.load(f)
  
  def getScore(self, prompt_config, path_ex, path_pred):
    
    exs = self.openJsonFile(path_ex)
    refs, src_infos = exs['tgt'], exs['src_info']

    preds = self.openJsonFile(path_pred)['hyps']

    assert len(preds) == len(refs)
      
    d_res = {}
    for metric in prompt_config['metrics']:

      if metric == "rouge":
        d_res = {**d_res, **self.computeRouge(preds, refs)}

      if metric == "bleu":
        d_res = {**d_res, **self.computeBleu(preds, refs)}

      if metric == "accuracy":
        d_res = {**d_res, **self.computeAcc(preds, refs)}
      
      if "constrain" in metric:
        d_res = {**d_res, **self.computeConstrain(preds, refs, src_infos, metric)}

    return d_res
        
  @staticmethod
  def openJsonFile(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data

  def computeRouge(self, preds, refs):
    
    rouge = load_metric("rouge")
    rouge.add_batch(predictions=preds, references=refs)
    d_res = rouge.compute()

    return {k:v.mid.fmeasure  for k, v in d_res.items()}

  def computeBleu(self, preds, refs):
    
    bleu = load_metric("bleu")
    bleu.add_batch(
        predictions=[pred.split() for pred in preds], 
        references=[[ref.split()] for ref in refs]
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


  def getAllScores(self, path_folder_preds, path_folder_data, init_from_sratch=False):
    
    if init_from_sratch == True:
      self.d_scores = {}

    evaluation_config = {**evaluation_new_tasks, **evaluation_T0evalsets}

    list_files = os.listdir(path_folder_preds)
    nb_files = len(list_files)
    for i, file in enumerate(list_files):
      
      if i % 50 == 0:
        print(f'{i}/{nb_files}')

      if "T0_3B" in file:
        evalset, eval_mode , prompt_name, model_name, _ = file.split('.')
        assert model_name == 'T0_3B'
        key = f'{model_name}.{evalset}.{eval_mode}.{prompt_name}'
      elif "sequencial" in file:
        evalset, eval_mode , prompt_name, model_size, rehearsal, _, model_name, _, model_from, step, _ = file.split('.')
        key = f'{model_name}.{rehearsal}.{step}.{evalset}.{eval_mode}.{prompt_name}.{model_from}'
      else:
        evalset, eval_mode , prompt_name, model_size, rehearsal, model_name, step, _ = file.split('.')
        key = f'{model_name}.{rehearsal}.{step}.{evalset}.{eval_mode}.{prompt_name}'

      if key in self.d_scores:
        continue 

      prompt_config = evaluation_config[evalset][eval_mode][prompt_name]
      path_pred = os.path.join(path_folder_preds, file)
      path_ex = os.path.join(path_folder_data, evalset, f'{prompt_name}.{eval_mode}.json')
      
      self.d_scores[key] = self.getScore(prompt_config, path_ex, path_pred)

    if self.path_dscores:
      with open(self.path_dscores, 'w') as f:
        json.dump(self.d_scores, f, indent=3)
        
        
def whatMetric(dataset_name, prompt_name, force_nlg='bleu', force_nlu='accuracy'):
  
  nlg_datasets = {'haiku', 'eli5', 'wiki_auto', 'gigaword'}
  nlu_datasets = { 'rte', 'copa', 'wic', 'winogrande', 'hellaswag', 'anli', 'cb', 'wsc', 'story_cloze', 'covidfact'}

  if 'constrain' in prompt_name:
    if 'constrain_start' in prompt_name:
      metric = 'start'
    elif 'constrain_contain' in prompt_name:
      metric = 'contain'
    elif 'constrain_end' in prompt_name:
      metric = 'end'

  elif dataset_name in nlg_datasets: 
    metric = force_nlg

  elif dataset_name in nlu_datasets: 
    metric = force_nlu
  else:
    raise NotImplementedError
     
  
  return metric


def get_color(group_name):

  color = None
  if group_name == 'T0_zero_shot_evalset':
    color = 'lime'
  elif group_name in {'wiki_auto', 'covidfact', 'gigaword_constrain'} :
    color = 'blue'
  elif group_name == 'gigaword':
    color = 'lightgreen'
  elif group_name == 'gigaword_start':
    color = 'bisque'
  elif group_name == 'gigaword_contain':
    color = 'lavender'
  elif group_name == 'gigaword_end':
    color = 'lightgrey'
  
  return color


def print_nicely(
    model_name, 
    d_datasets, 
    d_scores, 
    d_rehearsals,
    save_dir,
    force_nlg='bleu', 
    force_nlu='accuracy',
    do_normalise=True,
    get_color_custom=None
    ):
  
  
  if not get_color_custom:
    get_color_custom = get_color
   
  d_line_styles = {0: (0, (1, 10)), 250: (0, (5, 10)), 1000: 'solid'}
                   
  for group_name, group_datasets in d_datasets.items():
    for rehearsal, steps in d_rehearsals.items():
        scores = []
        for step in steps:
          step_scores = []
          for dataset_name, prompt_name, eval_mode in group_datasets:

            if step == 0:
              key = f'T0_3B.{dataset_name}.{eval_mode}.{prompt_name}'
            else:
              key = f'{model_name}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'

            step_scores.append(d_scores[key][whatMetric(dataset_name, prompt_name, force_nlg, force_nlu)])

          scores.append(sum(step_scores)/len(step_scores))

        if do_normalise:
          scores = [scores[i]/scores[0] for i in range(len(steps))]


        plt.plot(scores, label=f'{group_name}({rehearsal})', color=get_color_custom(group_name), linestyle=d_line_styles[rehearsal])

  plt.xticks(range(len(steps)), steps) #rotation='vertical')
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.title(f'{model_name}')
  plt.savefig(os.path.join(save_dir, f'{model_name}.{"normalized" if do_normalise else "absolute"}'), format='pdf')
  plt.show()
  return
