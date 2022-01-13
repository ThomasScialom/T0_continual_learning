import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F

class ApiT0():
  def __init__(self, name_or_path, batch_size=32, max_length=512, num_beams=1, is_cuda=True):
      self.name_or_path = name_or_path
      self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, low_cpu_mem_usage=True)
      self.num_beams = num_beams
      self.batch_size = batch_size
      self.max_length = max_length
      
      self.is_cuda = is_cuda
      if is_cuda == True:
        self.model.cuda()

  def predict(self, srcs):

    hyps = []
    for batch_i, idx in enumerate(range(0, len(srcs), self.batch_size)):
      
      if batch_i == 10:
        print(f'........batch {batch_i}/{len(srcs)//self.batch_size}')

      batch_srcs = srcs[idx:idx+self.batch_size]
      inputs = self.tokenizer(
        batch_srcs, 
        padding=True, 
        return_tensors="pt",
        max_length=self.max_length, 
        truncation=True
      )
      
      if self.is_cuda:
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()

      output_sequences = self.model.generate(
          input_ids=inputs['input_ids'],
          attention_mask=inputs['attention_mask'],
          do_sample=False,
          num_beams=self.num_beams,
      )
      
      hyps += self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    
    return hyps

  def predictBestOption(self, data, name_choice):  
    
    batch_size = self.batch_size//4

    srcs = data['src']

    d_tgts = {}
    if name_choice == "yes_no":
      d_tgts['yes'] = len(srcs)*['yes']
      d_tgts['no'] = len(srcs)*['no']
    elif name_choice == "option1_option2":
      d_tgts['option1'] = [ex["option1"] for ex in data['src_info']]
      d_tgts['option2'] = [ex["option2"] for ex in data['src_info']]
    elif name_choice == "choice1_choice2":
      d_tgts['choice1'] = [ex["choice1"] for ex in data['src_info']]
      d_tgts['choice2'] = [ex["choice2"] for ex in data['src_info']]

    d_loss = {}
    for name_tgt, tgts in d_tgts.items():
      
      list_loss = []
      for batch_i, idx in enumerate(range(0, len(srcs), batch_size)):

        encoding = self.tokenizer(
            srcs[idx:idx+batch_size], 
            padding='longest',
            truncation=True, 
            return_tensors="pt"
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    
        target_encoding = self.tokenizer(
            tgts[idx:idx+batch_size], 
            padding='longest', 
            truncation=True
        )
        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100

        if self.is_cuda:
          input_ids = input_ids.cuda()
          attention_mask = attention_mask.cuda()
          labels = labels.cuda()
      
        pred = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = pred.logits
        list_loss += [
          F.cross_entropy(
              logits[batch_i].view(-1, self.tokenizer.vocab_size), 
              labels[batch_i].view(-1)
          ).item() 
          for batch_i in range(batch_size)
        ]

      d_loss[name_tgt] = list_loss

    return d_loss

  def generateAll(self, path_src, path_hyp, choice=None):

    with open(path_src, 'r') as f:
      data = json.load(f)

    d_pred = {'hyps': self.predict(data['src'])}

    """
    usefull to check the class with the highest logit, but deactivate because we use only the raw text predicted
    if choice:
      d_loss = self.predictBestOption(data, choice)
      d_pred = {**d_pred, **d_loss}
    """

    with open(path_hyp, 'w') as f_w:
      json.dump(d_pred, f_w)     

    return

  
def generateAllPredictions(
  d_models, 
  d_datasets, 
  path_data, 
  path_predictions, 
  use_logs=True, 
  batch_size=32, 
  is_cuda=True
):

  for model_name in d_models:

    print(f'Loading {model_name}...')
    model = ApiT0(d_models[model_name], batch_size=batch_size, num_beams=1, is_cuda=is_cuda)
    print("...Loaded.")

    for dataset_name, d_prompt_modes in d_datasets.items():
      for eval_mode, prompt_modes in d_prompt_modes.items():
        for prompt_mode, d_prompt in prompt_modes.items():          
          path_src = os.path.join(path_data, dataset_name, f'{prompt_mode}.{eval_mode}.json')
          path_hyp = os.path.join(path_predictions, f'{dataset_name}.{eval_mode}.{prompt_mode}.{model_name}.json')
          print(f'Start predictions for: {path_src})...')
          if use_logs and os.path.exists(path_hyp):
            print("...predictions ALREADY done and use_logs==True, continue.")
            continue
          model.generateAll(path_src, path_hyp, d_prompt['choice'])
          print("...predictions done.")

    del model
