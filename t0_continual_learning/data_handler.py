import os
import json
import random
import csv
from typing import List
from collections import Counter
import urllib

from datasets import load_dataset
from promptsource.templates import DatasetTemplates

from t0_continual_learning import DIR

random.seed(666)


class PromptFormat():
  def __init__(self, config):
    
    self.config = config

    template_params = config['prompt_T0']['template_name']
    if template_params is not None:
      self.ag_news_prompts = DatasetTemplates(template_params[0], template_params[1])
  
  def filterDataset(self, dataset):
    
    return dataset
    
  def getDataset(self, eval_mode):
    
    hf_config = self.config['hf_dataset_config']
    dataset = load_dataset(
        path=hf_config['name'], 
        name=None if hf_config['option'] == "" else hf_config['option'], 
        split=eval_mode
    )
    dataset = self.filterDataset(dataset)
    
    return dataset
  
  def processAllExs(self, eval_mode, prompt_mode, prompt_name, limit_nb_examples=-1):
    
    dataset = self.getDataset(eval_mode)
    print('Total nb examples:', len(dataset))

    i = 0
    list_src_formated, list_tgt, list_ex = [], [], []
    for i in range(len(dataset)):
         
      ex = dataset[i]

      src, tgt = self.apply(ex, prompt_mode, prompt_name)
      list_tgt.append(tgt)
      list_src_formated.append(src)
      list_ex.append(ex)
      
      if i != -1 and i > limit_nb_examples:
        break
      if i % 5000 == 0:
        print("Examples processed:", i)
      
    return list_src_formated, list_tgt, list_ex

  def apply(self, ex, prompt_mode, prompt_name):
    
    if prompt_mode == 't0_template':
      is_random = prompt_name == "__RANDOM__"
      if is_random:
        all_template_names = self.ag_news_prompts.all_template_names
        prompt_name = random.choice(all_template_names)
      try:
        src_formated, tgt = self.ag_news_prompts[prompt_name].apply(ex)
      except:
        if is_random == False:
          raise NotImplemented(f"Prompt not correct: {prompt_name}")
        while all_template_names:
          failing_idx = all_template_names.index(prompt_name)
          all_template_names.pop(failing_idx)
          prompt_name = random.choice(all_template_names)
          output = self.ag_news_prompts[prompt_name].apply(ex)
          if len(output) == 2: # otherwise apply has failed
            break
        src_formated, tgt = output

    elif prompt_mode == 'custom':
      src_formated = self.applyCustom(ex, prompt_name)
      tgt = ex[self.config['hf_dataset_config']['tgt']]

    ex['prompt_name'] = prompt_name
    ex['prompt_mode'] = prompt_mode

    return src_formated, tgt

  def applyCustom(self, ex, prompt_name):
    
    prompt = random.choice(self.config['prompt_cutsom'][prompt_name])
    ex['prompt'] = prompt

    src_keys = self.config['hf_dataset_config']['src']

    if 'constrain' in prompt_name:
      src_keys += self.setConstrainAttr(ex, prompt_name, src_keys)
    
    for key in src_keys:
      prompt = prompt.replace(f'[{key}]', ex[key])
    
    return prompt

  def setConstrainAttr(self, ex, prompt_name, src_keys):
    
    if self.nb_tokens == None:
      nb_tokens = 1
      
    tgt = ex[self.config['hf_dataset_config']['tgt']]
    
    if 'start' in prompt_name:
      constrain_type = 'start'
      start_id = 0
    elif 'contain' in prompt_name:
      constrain_type = 'contain'
      start_id = random.randint(0, len(tgt.split())-1)    
    elif 'end' in prompt_name:
      constrain_type = 'end'
      start_id = len(tgt.split()) - 1
    else:
      raise(f"prompt_name not defined well: {prompt_name}")

    span_to_inserts = ' '.join(tgt.split()[start_id: start_id+nb_tokens])
    ex["TO_REPLACE_1"] = span_to_inserts
    ex["constrain_type"] = constrain_type

    additional_src_keys = ["TO_REPLACE_1"]
    return additional_src_keys


class ELI5promptFormat(PromptFormat):
  def __init__(self, config):
    super().__init__(config)

  def filterDataset(self, dataset):
    
    print('In clean')
    clean_exs = []
    for ex in dataset:
      
      question = ex['title']
      if not (question[0] in {'W', 'H'} and Counter(question)['?'] == 1):
        continue

      scores = ex['answers']['score']
      text = ex['answers']['text'][scores.index(max(scores))]
      
      clean_exs.append({'question': question, 'text': text})
    
    return clean_exs
  
  
class CovidFactPromptFormat(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)

  def getDataset(self, eval_mode):

    path = os.path.join(DIR, 'additional_datasets/covidfact', f'{eval_mode}.tsv')

    examples = []
    with open(path, 'r') as tsv_file:
      data = csv.reader(tsv_file, delimiter="\t")
      headers = next(data)
      headers = ['index', 'premise', 'hypothesis', 'entailment']
    
      for row in data:
        ex = {k: v for k, v in zip(headers, row)}
        ex['label'] = 1 if ex['entailment'] == 'entailment' else 0
        examples.append(ex)
    
    return examples
        
class WikiAutoPromptFormat(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)

  def getDataset(self, eval_mode):
    
    dataset = super(WikiAutoPromptFormat, self).getDataset('full')
    dataset = [ex for ex in dataset]
    if eval_mode == 'train':
      dataset = dataset[:-10000]
    if eval_mode == 'test':
      dataset = dataset[-4000:]
      
    return dataset

class StoryClozePromptFormat(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)

  def getDataset(self, eval_mode):
    
    hf_config = self.config['hf_dataset_config']
    dataset = load_dataset(
        path=hf_config['name'], 
        split=eval_mode,
        data_dir = os.path.join(DIR, 'additional_datasets/story_cloze/')
    )
    dataset = self.filterDataset(dataset)
          
    return dataset

  
class HaikuPromptFormat(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)

  def getDataset(self, eval_mode):
    
    with open(os.path.join(DIR, f'additional_datasets/haiku/haiku{eval_mode}.json'), 'r') as f:
      lines = f.readlines()

    dataset = [json.loads(line)["translation"] for line in lines]
          
    return dataset
  

class CovidQAPromptFormat(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)
  
  def filterDataset(self, dataset, limit=2500):
    
    clean_exs = []
    for ex in dataset:
      if ex["is_impossible"] == True:
        continue
      ex['tgt'] = ex['answers']['text'][0]
      ex['question'] = ex['question'][0].lower() + ex['question'][1:]
      clean_exs.append(ex)
        
    return clean_exs[:limit]
  
  
class RankSummary(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)

  def getDataset(self, eval_mode):
    
    list_batch_id = [3, 4, 5, 6, 7, 8, 9] if eval_mode == 'train' else [13]
    
    data = []
    for batch_id in list_batch_id:
      url = f'https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/comparisons/batch{batch_id}.json'

      response = urllib.request.urlopen(url)
      r = response.read().decode("utf-8") 
      data += [json.loads(l) for l in r.strip().split('\n')]
    
    clean_data = []
    for ex in data:
     
      clean_ex = {
        'post': ex['info']['post'],
        'sum0': ex['summaries'][0]['text'],
        'sum1': ex['summaries'][1]['text'],
        'label': str(ex['choice'])
      }
      clean_data.append(clean_ex)
          
    return clean_data
  
class ExplanationSNLI(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)
  
  def getDataset(self, eval_mode):
  
    name_batches = ['train_1', 'train_2'] if eval_mode == 'train' else [eval_mode]

    data = []
    for name_batch in name_batches:
      url = f'https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_{name_batch}.csv'

      response = urllib.request.urlopen(url)
      r = response.read().decode("utf-8") 
      lines = r.splitlines()
      reader = csv.reader(lines)
      parsed_csv = list(reader)

      header = parsed_csv[0]
      data += parsed_csv[1:]


    d_gold_label = {
      'entailment': 'entail each other',
      'contradiction': 'do not entail each other',
      'neutral': 'are unrelated',
    }

    clean_data = []
    for ex in data:

      clean_ex = {}
      for key, value in zip(header, ex):
        clean_ex[key] = value
      clean_ex['gold_label_prompt'] = d_gold_label[clean_ex['gold_label']]
      clean_ex['tgt'] = [v for k, v in clean_ex.items() if 'Explanation_' in k]
      clean_data.append(clean_ex)

    return clean_data    
  
  
class TwitterTop20(PromptFormat):
    
  """
  @article{tareaf2017r,
    title={R.: Tweets dataset-top 20 most followed users in Twitter social platform},
    author={Tareaf, Bin},
    journal={Harvard Dataverse},
    volume={2},
    year={2017},
    url={https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JBXKFD/F4FULO}
  }
  """
  
  def __init__(self, config):
    super().__init__(config)
  
  def getDataset(self, eval_mode):
  
    with open(os.path.join(DIR, f'additional_datasets/twitterTop20/tweets.csv'), 'r') as f:
      spamreader = csv.reader(f, delimiter=',')
      parsed_csv = list(spamreader)

      header = parsed_csv[0]
      data = parsed_csv[1:]

    clean_data = []
    for ex in data:

      d_ex = {k:v for k, v in zip(header, ex)}
      
      # we want to select only hashtag's tweets for our dataset to add this token to the prompt's instruction
      if '#' not in d_ex['content'] or len(d_ex['content'].split('#')[1].split()) < 2:
        continue

      hashtag_token = '#' + d_ex['content'].split('#')[1].split()[0]
      d_ex["hashtag_token"] = hashtag_token

      clean_data.append(d_ex)
    
    
    # Create the same split consistantly between test/val/train
    random.seed(666)
    random.shuffle(clean_data)

    if eval_mode == 'test':
      clean_data = clean_data[:250]
    elif eval_mode == 'validation':
      clean_data = clean_data[250:500]
    elif eval_mode == 'train':
      clean_data = clean_data[500:]
    else:
      raise NotImplementedError(eval_mode)

    return clean_data
    
    
    
      
class DialogueEmpathic(PromptFormat):
  
  def __init__(self, config):
    super().__init__(config)
  
  def filterDataset(self, dataset):
  
    clean_data = []

    temp = []
    clean_ex = None
    for ex in dataset:

      next_idx, next_speaker_idx = ex['utterance_idx'], ex['speaker_idx']
      if clean_ex == None or clean_ex['conv_id'] != ex['conv_id']:

        if clean_ex:
          clean_data.append(clean_ex)

        if next_idx != 1:
          continue

        clean_ex  = {
          'conv_id': ex['conv_id'],
          'context': ex['context'],
          'prompt': ex['prompt'],
          'input_texts': [ex['utterance']]
        }

        utterance_idx, speaker_idx = next_idx, next_speaker_idx
        continue

      if speaker_idx == ex['speaker_idx'] or next_idx != utterance_idx + 1:
        temp.append(ex['conv_id'])
        continue
      assert speaker_idx != ex['speaker_idx']
      assert next_idx == utterance_idx + 1
      utterance_idx, speaker_idx = next_idx, next_speaker_idx

      clean_ex['input_texts'].append(ex['utterance'])

    clean_data.append(clean_ex)

    examples = []
    for d_conv in clean_data:

      input_texts = [text.replace('_comma_', ',') for text in d_conv['input_texts']]
      prompt = d_conv['prompt'].replace('_comma_', ',')
      for i in range(1, len(input_texts)):

        ex = {
            'context_emotion': d_conv['context'],
            'context_prompt': prompt,
            'conv_id': d_conv ['conv_id'],
            'idx': i,
            'input_text': '\n'.join(['- '+text for text in input_texts[:i]]),
            'output_text': input_texts[i]
        }
        examples.append(ex)

    return examples

    
# utils functions 
def write_data(srcs, tgts, src_infos, final_folder, prompt_name, eval_mode):
  if not os.path.exists(final_folder):
    os.mkdir(final_folder)

  output_path = os.path.join(final_folder, f'{prompt_name}.{eval_mode}.json')
  with open(output_path, 'w') as f:
    json.dump({'src': srcs, 'tgt': tgts, 'src_info':src_infos}, f)


def process_datasets(d_datasets, limit_nb_examples, path_data="data"):
  
  for dataset_name, dataset_modes  in d_datasets.items():
    
    with open(os.path.join(DIR, 'configs', f'{dataset_name}.json'), 'r') as f:
      config = json.load(f)
    
    if dataset_name == 'eli5':
      promptFormat = ELI5promptFormat(config)
    elif dataset_name == 'covidfact':
      promptFormat = CovidFactPromptFormat(config)
    elif dataset_name == 'wiki_auto':
      promptFormat = WikiAutoPromptFormat(config)
    elif dataset_name == 'story_cloze':
      promptFormat = StoryClozePromptFormat(config)
    elif dataset_name == 'haiku':
      promptFormat = HaikuPromptFormat(config)
    elif dataset_name == 'covid_qa_deepset':
      promptFormat = CovidQAPromptFormat(config)
    elif dataset_name == 'rank_summary':
      promptFormat = RankSummary(config)
    elif dataset_name == 'eSNLI':
      promptFormat = ExplanationSNLI(config)
    elif dataset_name == 'twitter_top20':
      promptFormat = TwitterTop20(config)
    elif dataset_name == 'empathetic_dialogues':
      promptFormat = DialogueEmpathic(config)
      
    else:
      promptFormat = PromptFormat(config)

    for eval_mode, prompt_modes in dataset_modes.items():
      for prompt_mode, prompt_name in prompt_modes:

        print('Starting for', dataset_name, eval_mode, prompt_mode, prompt_name, '...')
        random.seed(666)
        srcs, tgts, src_infos = promptFormat.processAllExs(eval_mode, prompt_mode, prompt_name, limit_nb_examples)
        
        final_folder = os.path.join(path_data, dataset_name)
        write_data(srcs, tgts, src_infos, final_folder, prompt_name, eval_mode)

        print('... done.')

  return


# formating the data to the training translation format, including reharsal data

import os
import json
import random
from t0_continual_learning.config_variables import t0_train_datasets

def buildReharsalDataset(config_name, list_config_reharsals, rehearsal_number, path_data, percentage=1):
  """
  Note that for rehearsal proportion, we do as follow:
  - the main dataset has a number of examples corresponding to the sum of exs/prompts, as defined in config_reharsal['new_dataset']['prompts']
  - for each dataset in config_reharsal['reharsal']['list_datasets'], we will add a number of examples N with N = reharsal_number * factor_sample
    where factor_sample is 1 by default if not defined specifically in the config_reharsal
    
  In addition, we multiply the overall numbers by *percentage* which allows to have small validation files to evaluate during training on e.g. 0.1% of the training data.
  """
  
  random.seed(666)

  def doFormat(s, t, d): 
    if isinstance(t, List):
      t = t[0]
    return json.dumps({"translation": {"en1": s, "en2": t}}, ensure_ascii=False)

  def sampleFromPath(path_complete, k):
    with open(path_complete, 'r') as f:
        data = json.load(f)
    
    nb_total = len(data["src"])
    list_idx = random.choices(range(nb_total), k=k)

    for i, idx in enumerate(list_idx):
      list_output.append(
          doFormat(data["src"][idx], data["tgt"][idx], dataset_name)
      )

    return 

  list_output = []
  isFistExp = True
  exp_name = config_name
  while exp_name:
    print(f"Starting sampling {exp_name}")

    config_dict = list_config_reharsals[exp_name]
    dataset_name = config_dict['new_dataset']['name']
    eval_mode = config_dict['new_dataset']['eval_mode']
    
    for prompt, nb_to_sample in config_dict['new_dataset']['prompts'].items():
      if isFistExp == False:
        nb_to_sample = rehearsal_number
      path_complete = os.path.join(path_data, dataset_name, f'{prompt}.{eval_mode}.json')
      sampleFromPath(path_complete, int(nb_to_sample*percentage))

    exp_name = config_dict['reharsal']['inheritFrom']
    isFistExp = False

  prompt, eval_mode = '__RANDOM__', 'train'
  for dataset_name in t0_train_datasets:
    path_complete = os.path.join(path_data, dataset_name, f'{prompt}.{eval_mode}.json')
    sampleFromPath(path_complete, int(rehearsal_number*percentage))

  print('Sampling completed. Now shuffling')
  random.shuffle(list_output)
  
  return list_output
  
def format2train(config_name, list_config_reharsals, rehearsal_number, path_data, final_folder):

  for mode, percentage in zip(['validation', 'train'], [0.03, 1]):
    print(mode, '.....')
    list_output = buildReharsalDataset(config_name, list_config_reharsals, rehearsal_number, path_data, percentage=percentage)
    with open(os.path.join(final_folder, f"{mode}.{config_name}.continual{rehearsal_number}.json"), "w") as f_w:
      for line in list_output:
        f_w.write(line + "\n")

  return list_output
