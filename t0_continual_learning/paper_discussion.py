import json
import random
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from transformers import pipeline

from t0_continual_learning import model_prediction

def createMutltipleConstrain(GLOBAL_PATH):

  random.seed(666)
  path = f'{GLOBAL_PATH}/data/gigaword/make_a_title.test.json'
  with open(path, 'r') as f:
    data = json.load(f)

  d_srcs = {1: [], 2: [], 3: [], 'src_info': [], 'tgt': []}
  for tgt, ex in zip(data['tgt'], data['src']):
    
    if len(tgt.split()) < 4:
      continue

    start_ids = random.sample(range(len(tgt.split())-1) , k=3) 
    tokens = [tgt.split()[i] for i in start_ids]
    d_srcs['src_info'].append({'tokens': tokens})
    d_srcs['tgt'].append(tgt)

    for i in range(1, 4):
      if i == 1:
        replacement = f'Make a title for this article, containing "{tokens[0]}"'
      if i == 2:
        replacement = f'Make a title for this article, containing "{tokens[0]}" and "{tokens[1]}"'
      if i == 3:
        replacement = f'Make a title for this article, containing "{tokens[0]}", "{tokens[1]}" and "{tokens[2]}"'
      new_ex = ex.replace('Make a title for this article', replacement)
      d_srcs[i].append(new_ex)
    

  for i in range(1, 4):
    final_path = path = f'{GLOBAL_PATH}/data/_analysis/mutliple_constrain_{i}.json'
    with open(final_path, 'w') as f:
      json.dump({'src': d_srcs[i], 'tgt': d_srcs['tgt'], 'src_info': d_srcs['src_info']}, f)

def createTwitterConstrain(GLOBAL_PATH):

  random.seed(666)
  path = f'{GLOBAL_PATH}/data/twitter_top20/tweet_as+about.test.json'
  with open(path, 'r') as f:
    data = json.load(f)

  d_srcs = {'src': [], 'src_info': [], 'tgt': []}
  for tgt, ex in zip(data['tgt'], data['src']):
    
    tgt_split_no_hash = [tok for tok in tgt.split() if '#' not in tok and 'http' not in tok]
    if len(tgt_split_no_hash) < 2:
      continue

    start_ids = random.sample(range(len(tgt_split_no_hash)-1) , k=1) 
    tokens = [tgt_split_no_hash[i] for i in start_ids]
    d_srcs['src_info'].append({'tokens': tokens})
    d_srcs['tgt'].append(tgt)

    new_ex = ex + f', containing "{tokens[0]}"'
    d_srcs['src'].append(new_ex)
    

  final_path = path = f'{GLOBAL_PATH}/data/_analysis/twitter_constrain.json'
  with open(final_path, 'w') as f:
    json.dump(d_srcs, f)


def createEmotionHaiku(GLOBAL_PATH, nbEx=500):

  random.seed(666)
  path = f'{GLOBAL_PATH}/data/haiku/do_nothing.test.json'
  with open(path, 'r') as f:
    data = json.load(f)

  for emotion in ["faithful", "grateful", "trusting", "content", "sad", "lonely", "angry", "terrified", "nostalgic"]:

    d_srcs = {'src': [], 'src_info': [], 'tgt': []}
    for tgt, ex in zip(data['tgt'][:nbEx], data['src'][:nbEx]):

      d_srcs['src_info'].append({'emotion': emotion})
      d_srcs['tgt'].append(tgt)

      new_ex = ex +  f' The associated emotion is "{emotion}".'
      d_srcs['src'].append(new_ex)
    
    final_path = path = f'{GLOBAL_PATH}/data/_analysis/haiku_emotion_{emotion}.json'
    with open(final_path, 'w') as f:
      json.dump(d_srcs, f)

def predictAnalysis(GLOBAL_PATH, is_cuda=True):
  
  name = f'{GLOBAL_PATH}/models/sequential/11B.rehersal1000.sequencial.twitter_top20.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5->empathetic_dialogues->eSNLI_143'    
  model = model_prediction.ApiT0(name, is_cuda=is_cuda)

  path = f'{GLOBAL_PATH}/data/_analysis'
  for file in os.listdir(path):
    with open(os.path.join(path, file), 'r') as f:
      data = json.load(f)

    outputs = model.predict(data['src'])

    final_path = path 
    with open(os.path.join(path, 'pred.' + file), 'w') as f:
      json.dump({'hyp': outputs}, f)

def predictBaselines(GLOBAL_PATH, is_cuda = True):

  paths = [
          (f'{GLOBAL_PATH}/data/_analysis/pred.twitter_constrain.baseline.json', f'{GLOBAL_PATH}/data/twitter_top20/tweet_as+about.test.json'),
          (f'{GLOBAL_PATH}/data/_analysis/pred.mutliple_constrain.baseline.json', f'{GLOBAL_PATH}/data/gigaword/make_a_title.test.json')
  ]

  name = f'{GLOBAL_PATH}/models/sequential/3B.rehersal1000.sequencial.twitter_top20.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5->empathetic_dialogues->eSNLI_143'    
  model = model_prediction.ApiT0(name, is_cuda=is_cuda)

  for pred_path, src_path in paths: 

    print(pred_path)
    with open(src_path, 'r') as f:
      data = json.load(f)
    
    outputs = model.predict(data['src'])
    
    with open(pred_path, 'w') as f:
      json.dump({'hyp': outputs}, f)
    



def getConstrainResults(GLOBAL_PATH):
  
  random.seed(1)

  formating = lambda x: round(100*x, 1)

  d_scores = {
      'Random': {},
      'CT0': {},
    } 
  
    
  def getData(path):

    def getSkippedIds(path, k):
      
      with open(path, 'r') as f:
        data = json.load(f)
      skipped_ids = []
      for i, tgt in enumerate(data['tgt']):
        
        nb_toks = len(tgt.split())
        if 'twitter' in path:
          nb_toks = len([tok for tok in tgt.split() if '#' not in tok and 'http' not in tok])
        
        if nb_toks < k:
          skipped_ids.append(i)
      return set(skipped_ids)

    if 'twitter_constrain' in path:
        path_baseline = f'{GLOBAL_PATH}/data/_analysis/pred.twitter_constrain.baseline.json'
        skipped_ids = getSkippedIds(f'{GLOBAL_PATH}/data/twitter_top20/tweet_as+about.test.json', 2)
    elif 'mutliple_constrain' in path:
        path_baseline = f'{GLOBAL_PATH}/data/_analysis/pred.mutliple_constrain.baseline.json'
        skipped_ids = getSkippedIds(f'{GLOBAL_PATH}/data/gigaword/make_a_title.test.json', 4)

    with open(path_baseline, 'r') as f:
      data = json.load(f)
      baselines = [h for i, h in enumerate(data['hyp']) if i not in skipped_ids]

    with open(path, 'r') as f:
      data = json.load(f)
      list_tokens = [ex['tokens'] for ex in data['src_info']]

    with open(path.replace('_analysis/', '_analysis/pred.'), 'r') as f:
      data = json.load(f)
      hyps = data['hyp']
    
    assert len(hyps) == len(list_tokens)
    assert len(hyps) == len(baselines)

    return baselines, hyps, list_tokens

  def getScore(hyps, baselines, prompt_tokens, i=3):

    baseline_scores, ct0_scores = [], []
    for hyp, baseline, prompt_tokens in zip(hyps, baselines, list_prompt_tokens):

      ct0_scores.append(sum([tok in hyp for tok in prompt_tokens[:i]]) == i)
      baseline_scores.append(sum([tok in baseline for tok in prompt_tokens[:i]]) == i)

    ct0_scores = formating(np.average(ct0_scores))
    baseline_scores = formating(np.average(baseline_scores))

    return ct0_scores, baseline_scores

  for i in range(1, 4):
    
    path = f'{GLOBAL_PATH}/data/_analysis/mutliple_constrain_{i}.json'
    baselines, hyps, list_prompt_tokens = getData(path)
    ct0_scores, baseline_scores = getScore(hyps, baselines, list_prompt_tokens, i=i)
    
    d_scores['CT0'][i] = ct0_scores
    d_scores['Random'][i] = baseline_scores


  path = f'{GLOBAL_PATH}/data/_analysis/twitter_constrain.json'
  baselines, hyps, list_prompt_tokens = getData(path)
  ct0_scores, baseline_scores = getScore(hyps, baselines, list_prompt_tokens, i=1)
  d_scores['CT0']['Twitter'] = ct0_scores
  d_scores['Random']['Twitter'] = baseline_scores

  return d_scores

def getHaikuEmotionResults(GLOBAL_PATH):
  """
  return d_labels = {
    'sad': 0.404,
    'terrified': 0.418,
    'angry': 0.428,
    'lonely': 0.482,
    'trusting': 0.666,
    'content': 0.762,
    'faithful': 0.768,
    'grateful': 0.824,
  } 
  """
  sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

  d_labels = {}
  for emotion in ['sad', 'terrified', 'angry', 'lonely', 'trusting', 'content', 'faithful', 'grateful']:
    print(emotion)
    
    path = f'{GLOBAL_PATH}/data/_analysis/haiku_emotion_{emotion}.json'
    with open(path.replace('_analysis/', '_analysis/pred.'), 'r') as f:
      data = json.load(f)
      hyps = data['hyp']

      labels = sentiment_analysis(hyps)
      d_labels[emotion] = [label['label'] for label in labels]

  for k, labels in d_labels.items():
    d_labels[k] = sum([label=='POSITIVE' for label in labels])/len(labels)

  return d_labels    

def printDf(d_res):
  df = pd.DataFrame(d_res)
  df = df.transpose() 
  return df

def GetFig(d_labels, save_dir):

  cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", 'yellow', "green"])
  color = cmap([v/max(d_labels.values()) for v in d_labels.values()])

  plt.bar(d_labels.keys(), d_labels.values(), color=color)
  plt.xticks(rotation = 45)
  plt.ylabel('% Positive')
  plt.savefig(save_dir, format='pdf', bbox_inches='tight')
  plt.show()
  
