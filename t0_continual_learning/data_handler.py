import os
import json
import random

from datasets import load_dataset
from promptsource.templates import DatasetTemplates

random.seed(666)


class PromptFormat():
  def __init__(self, config):
    
    self.config = config

    template_params = config['prompt_T0']['template_name']
    if template_params is not None:
      self.ag_news_prompts = DatasetTemplates(template_params[0], template_params[1])
  
  def processAllExs(self, eval_mode, prompt_mode, prompt_name, limit_nb_examples=-1):

    hf_config = self.config['hf_dataset_config']
    dataset = load_dataset(
        path=hf_config['name'], 
        name=None if hf_config['option'] == "" else hf_config['option'], 
        split=eval_mode
    )
    
    print('Total nb examples:', len(dataset))

    i = 0
    list_src_formated, list_tgt, list_ex = [], [], []
    for i in range(len(dataset)):
      
      if i != -1 and i > limit_nb_examples:
        break
      if i % 5000 == 0:
        print("Examples processed:", i)
      
      ex = dataset[i]

      src, tgt = self.apply(ex, prompt_mode, prompt_name)
      list_tgt.append(tgt)
      list_src_formated.append(src)
      list_ex.append(ex)
      
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


def write_data(srcs, tgts, src_infos, final_folder, prompt_name, eval_mode):
  if not os.path.exists(final_folder):
    os.mkdir(final_folder)

  output_path = os.path.join(final_folder, f'{prompt_name}.{eval_mode}.json')
  with open(output_path, 'w') as f:
    json.dump({'src': srcs, 'tgt': tgts, 'src_info':src_infos}, f)


def process_datasets(d_datasets, limit_nb_examples, path=""):
  
  for dataset_name, dataset_modes  in d_datasets.items():
    
    with open(os.path.join(path, f'configs/{dataset_name}.json'), 'r') as f:
      config = json.load(f)  
    promptFormat = PromptFormat(config)

    for eval_mode, prompt_modes in dataset_modes.items():
      for prompt_mode, prompt_name in prompt_modes:

        print('Starting for', dataset_name, eval_mode, prompt_mode, prompt_name, '...')
        random.seed(666)
        srcs, tgts, src_infos = promptFormat.processAllExs(eval_mode, prompt_mode, prompt_name, limit_nb_examples)
        
        final_folder = os.path.join(path, f'/data/{dataset_name}')
        write_data(srcs, tgts, src_infos, final_folder, prompt_name, eval_mode)

        print('... done.')

  return
