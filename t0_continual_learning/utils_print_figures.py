import matplotlib.pyplot as plt
import os

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


def whatMetricDefault(dataset_name, prompt_name, default_nlg='bleu', default_nlu='accuracy'):
  
  nlg_datasets = {'haiku', 'eli5', 'wiki_auto', 'gigaword', 'covid_qa_deepset', 'empathetic_dialogues', 'twitter_top20', 'eSNLI'}
  nlu_datasets = { 'rte', 'copa', 'wic', 'winogrande', 'hellaswag', 'anli', 'cb', 'wsc', 'story_cloze', 'covidfact', 'rank_summary'}

  if 'constrain' in prompt_name:
    if 'constrain_start' in prompt_name:
      metric = 'start'
    elif 'constrain_contain' in prompt_name:
      metric = 'contain'
    elif 'constrain_end' in prompt_name:
      metric = 'end'
  
  elif dataset_name == 'asset': 
    metric = 'sari'
  elif dataset_name == 'haiku': 
    metric = 'eq_weighted'
  elif dataset_name == 'eli5': 
    metric = 'jensenFirstToken'
   
  elif dataset_name in nlg_datasets: 
    metric = default_nlg

  elif dataset_name in nlu_datasets: 
    metric = default_nlu
  
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


def getScoresSequencial(d_scores, models, config_evaluation, model_size, default_nlg='bleu', default_nlu='accuracy', whatMetric=None):
  
  if not whatMetric:
    whatMetric = whatMetricDefault
    
    
  scores = {}

  for group_name, group_datasets in config_evaluation.items(): 

    last_step = 0
    all_steps = []
    scores[group_name] = []
    
    for model_name, model_from, rehearsal, steps in models:
      for step in steps:
        all_steps.append(last_step + step)

        step_score = 0
        for dataset_name, prompt_name, eval_mode in group_datasets['list_dataset']:
          # calculate the score
          if model_name == 'T0_3B' or model_name == 'T0pp':
            key = f'{model_name}.{dataset_name}.{eval_mode}.{prompt_name}'
          elif model_from == None:
            key = f'{model_name}.{model_size}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'
          else:
            key = f'{model_name}.{model_size}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}.{"->".join(model_from)}'

          nlg_metric = default_nlg if dataset_name != 'wiki_auto' else 'sari'
          if key not in d_scores:
            print(f'Break for {group_datasets["list_dataset"]}: {key} does not exist')
            step_score = None
            break
          step_score += d_scores[key][whatMetric(dataset_name, prompt_name, nlg_metric, default_nlu)]
        
        if step_score is not None:
            step_score = step_score/len(group_datasets['list_dataset'])
        scores[group_name].append(step_score)
      
      last_step += step
      
  return scores, all_steps

def printSequencialFigure(d_scores, models, config_evaluation, save_dir, 
                          model_size='3B', do_normalise=True, 
                          default_nlg='bleu', default_nlu='accuracy', 
                          whatMetric=None
                         ):
  

  scores, all_steps = getScoresSequencial(
      d_scores, 
      models, 
      config_evaluation, 
      model_size=model_size,
      default_nlg=default_nlg, 
      default_nlu=default_nlu, 
      whatMetric=whatMetric
  )
  
  
  plt.figure(figsize=(8, 6))
  plt.xlabel('Steps')
  plt.ylabel('Relative Gain')

  for k_name, v_scores in scores.items():  

    if do_normalise:
        #v_scores = [v_scores[i]/v_scores[0]-1 for i in range(len(v_scores))]
        v_scores = [v_scores[i]/v_scores[config_evaluation[k_name]['last_train']] 
                    if v_scores[i] else None  for i in range(len(v_scores))]


    x = config_evaluation[k_name]['steps']
    y = [v_scores[i] for i in config_evaluation[k_name]['steps']]

    plt.plot(x, y, label=k_name)

  plt.xticks(range(len(all_steps)), all_steps, rotation='vertical')
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  if save_dir is not None:
    plt.savefig(os.path.join(save_dir, '->'.join(models[-1][1] + [models[-1][0]])), format='pdf', bbox_inches='tight')
  plt.show()
  return scores
  
  
def printNonSequencialFigure(
    model_name, 
    d_datasets, 
    d_scores, 
    d_rehearsals,
    save_dir,
    model_size='3B',
    default_nlg='bleu', 
    default_nlu='accuracy',
    do_normalise=True,
    get_color_custom=None,
    ):
  
  
  if not get_color_custom:
    get_color_custom = get_color
   
  d_line_styles = {0: (0, (1, 10)), 250: (0, (5, 10)), 1000: 'solid'}
  
  d_scores_fig = {}
  for group_name, group_datasets in d_datasets.items():
    for rehearsal, steps in d_rehearsals.items():
        scores = []
        for step in steps:
          step_scores = []
          for dataset_name, prompt_name, eval_mode in group_datasets:

            if step == 0:
              baseline_name = 'T0_3B' if model_size == '3B' else 'T0pp'
              key = f'{baseline_name}.{dataset_name}.{eval_mode}.{prompt_name}'
            else:
              key = f'{model_name}.{model_size}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'

            step_scores.append(d_scores[key][whatMetric(dataset_name, prompt_name, default_nlg, default_nlu)])

          scores.append(sum(step_scores)/len(step_scores))
        
        d_scores_fig[f'{group_name}.{rehearsal}'] = scores
        if do_normalise:
          scores = [scores[i]/scores[0]-1 for i in range(len(steps))]
        
        plt.plot(scores, label=f'{group_name}({rehearsal})', color=get_color_custom(group_name), linestyle=d_line_styles[rehearsal])

  plt.xticks(range(len(steps)), steps) #rotation='vertical')
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.title(f'{model_name}')
  plt.savefig(os.path.join(save_dir, f'{model_name}.{"normalized" if do_normalise else "absolute"}'), format='pdf')
  plt.show()
  return d_scores_fig
  
 
