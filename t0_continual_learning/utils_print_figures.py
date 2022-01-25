import matplotlib.pyplot as plt

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


def getScoresSequencial(d_scores, models, config_evaluation, force_nlg='bleu', force_nlu='accuracy'):
  
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
          if model_name == 'T0_3B':
            key = f'T0_3B.{dataset_name}.{eval_mode}.{prompt_name}'
          elif model_from == None:
            key = f'{model_name}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'
          else:
            key = f'{model_name}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}.{"->".join(model_from)}'

          nlg_metric = force_nlg if dataset_name != 'wiki_auto' else 'sari'
          step_score += d_scores[key][whatMetric(dataset_name, prompt_name, nlg_metric, force_nlu)]
        scores[group_name].append(step_score/len(group_datasets))
      
      last_step += step
      
  return scores, all_steps

def printSequencialFigure(d_scores, models, config_evaluation, save_dir, do_normalise=True):
  
  scores, all_steps = getScoresSequencial(d_scores, models, config_evaluation)

  plt.figure(figsize=(8, 6))
  plt.xlabel('Steps')
  plt.ylabel('Relative Gain')

  for k_name, v_scores in scores.items():  

    if do_normalise:
        v_scores = [v_scores[i]/v_scores[0]-1 for i in range(len(v_scores))]

    x = config_evaluation[k_name]['steps']
    y = [v_scores[i] for i in config_evaluation[k_name]['steps']]

    plt.plot(x, y, label=k_name)

  plt.xticks(range(len(all_steps)), all_steps, rotation='vertical')
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.savefig(os.path.join(save_dir, '->'.join(models[-1][1] + [models[-1][0]])), format='pdf')
  plt.show()
  
  
  
  def printNonSequencialFigure(
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
  
  
