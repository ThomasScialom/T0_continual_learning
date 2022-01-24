import os
import json
import random
import csv
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
    
    tgt = ex[self.config['hf_dataset_config']['tgt']]
    
    nb_tokens = 1
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

def buildReharsalDataset(config_reharsal, reharsal_datasets, path_data, reharsal_number, percentage=1):
  
  list_output = []

  doFormat = lambda s, t, d: json.dumps(
      {"translation": {"en1": s, "en2": t}}
      , ensure_ascii=False
      )

  def sampleFromPath(path_complete):
    with open(path_complete, 'r') as f:
        data = json.load(f)
    nb_total = len(data["src"])
    random.seed(666)
    list_idx = random.choices(range(nb_total), k=int(nb_to_sample*percentage))

    for i, idx in enumerate(list_idx):
      list_output.append(
          doFormat(data["src"][idx], data["tgt"][idx], dataset_name)
          )

    return 

  
  dataset_name = config_reharsal['new_dataset']['name']
  eval_mode = config_reharsal['new_dataset']['eval_mode']
  dataset_prompts = config_reharsal['new_dataset']['prompts']
  print(f"Starting sampling {dataset_name}")
  for prompt, nb_to_sample in dataset_prompts.items():
    path_complete = os.path.join(path_data, dataset_name, f'{prompt}.{eval_mode}.json')
    sampleFromPath(path_complete)

  dataset_names = config_reharsal['reharsal']['list_datasets']
  nb_to_sample = reharsal_number
  print(f"Starting sampling reharsal ({len(dataset_names)} datasets), nb_to_sample={nb_to_sample}")
  for dataset_name in dataset_names:
    for path_complete in os.listdir(os.path.join(path_data, dataset_name)):
      if '__RANDOM__' not in path_complete:
        continue
      sampleFromPath(os.path.join(path_data, dataset_name, path_complete))
  
  print('Sampling completed. Now shuffling')
  random.shuffle(list_output)
  
  return list_output
  
def format2train(config_reharsal, reharsal_datasets, path_data, reharsal_number):
  
  final_folder = os.path.join(path_data, '_training_files')
  if not os.path.exists(final_folder):
    os.mkdir(final_folder)
    
  list_output = buildReharsalDataset(config_reharsal, reharsal_datasets, path_data, reharsal_number)
  with open(os.path.join(final_folder, f"train.{config_reharsal['name_exp']}.continual{reharsal_number}.json"), "w") as f_w:
    for line in list_output:
      f_w.write(line + "\n")
      
  list_output = buildReharsalDataset(config_reharsal, reharsal_datasets, path_data, reharsal_number, percentage=0.03)
  with open(os.path.join(final_folder, f"validation.{config_reharsal['name_exp']}.continual{reharsal_number}.json"), "w") as f_w:
    for line in list_output:
      f_w.write(line + "\n")

  return list_output
