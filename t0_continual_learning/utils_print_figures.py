import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import os
from t0_continual_learning import metric_scorer
from t0_continual_learning.config_variables import evaluation_new_tasks, evaluation_T0evalsets

class SeqFormating():

  def __init__(
      self, 
      list_path_folder_preds, path_folder_data, path_dscores=None, 
      d_map_tasks=None,
      whatMetric=None, 
      d_map_metrics=None, 
      modelNames=None, 
      model_size='3B', 
      print_detail_T0=False
    ):
    
    print('Loading the metricScorer and updating with predictions files')
    metricScorer = metric_scorer.MetricScorer(path_dscores)
    for path_folder_preds in list_path_folder_preds:
      metricScorer.getAllScores(path_folder_preds, path_folder_data)

    self.metricScorer = metricScorer
    self.d_scores = metricScorer.d_scores
    print('MetricScorer completed. Ready to print tables and figures.')

    self.print_detail_T0 = print_detail_T0
    self.model_size = model_size

    self.list_zero_shot = [
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

    self.loadMapNameDict(d_map_metrics, whatMetric, modelNames, d_map_tasks)

    self.loadNonSeqConfigs()
    self.loadSeqConfigs()

  def setModelSize(self, model_size):
    self.model_size = model_size
    self.loadNonSeqConfigs()
    self.loadSeqConfigs()

  def loadNonSeqConfigs(self):
    
    self.non_seq_d_rehearsals = {
      0: [0, 40, 80, 97], 
      250: [0, 40, 80, 106], 
      1000: [0, 40, 80, 120, 134], 
    }

    self.non_seq_config_images = {
      'wiki_auto': {
          'wiki_auto': [('wiki_auto', 'simplification_1', 'test')],
          'T0_zero_shot_evalset': self.list_zero_shot
      },
      'gigaword': {
          'gigaword_constrain': [('gigaword', 'constrain_start+make_a_title', 'test'),
                                ('gigaword', 'constrain_contain+make_a_title', 'test'),
                                ('gigaword', 'constrain_end+make_a_title', 'test')],
          #'gigaword_start': [('gigaword', 'constrain_start+make_a_title', 'test')],
          #'gigaword_contain': [('gigaword', 'constrain_contain+make_a_title', 'test')],
          #6'gigaword_end': [('gigaword', 'constrain_end+make_a_title', 'test')],
          #'gigaword': [('gigaword', 'make_a_title', 'test')],
          'T0_zero_shot_evalset': self.list_zero_shot
      },
      'haiku': {
          'haiku': [('haiku', 'do_nothing', 'test')],
          'T0_zero_shot_evalset': self.list_zero_shot
      },
    }

  def loadSeqConfigs(self):

    self.models = None
    if self.model_size == '3B':
      self.models = [
              ('T0_3B', None, 1000, [0]),
              ('wiki_auto', None, 1000, [40, 80, 120, 134]), #
              ('gigaword', ['wiki_auto'], 1000, [40, 80, 120, 137]),
              ('haiku', ['wiki_auto', 'gigaword'], 1000, [40, 80, 120, 137 ]),
              ('covid_qa_deepset', ['wiki_auto', 'gigaword', 'haiku'], 1000, [40, 80, 120, 139]),
              ('eli5', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset'], 1000, [40, 80, 120, 140]),
              ('empathetic_dialogues', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5'], 1000, [40, 80, 120, 141]),
              ('eSNLI', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5', 'empathetic_dialogues'], 1000, [40, 80, 120, 142]),
              ('twitter_top20', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5', 'empathetic_dialogues', 'eSNLI'], 1000, [40, 80, 120, 143]),
      ]
    elif self.model_size == '11B':
      self.models = [
              ('T0_3B', None, 1000, [0]),
              ('wiki_auto', None, 1000, [40, 80, 120, 134]), #
              ('gigaword', ['wiki_auto'], 1000, [40, 80, 120, 134]),
              ('haiku', ['wiki_auto', 'gigaword'], 1000, [40, 80, 120, 138 ]),
              ('covid_qa_deepset', ['wiki_auto', 'gigaword', 'haiku'], 1000, [40, 80, 120, 139]),
              ('eli5', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset'], 1000, [40, 80, 120, 140]),
              ('empathetic_dialogues', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5'], 1000, [40, 80, 120, 141]),
              ('eSNLI', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5', 'empathetic_dialogues'], 1000, [40, 80, 120, 142]),
              ('twitter_top20', ['wiki_auto', 'gigaword', 'haiku', 'covid_qa_deepset', 'eli5', 'empathetic_dialogues', 'eSNLI'], 1000, [40, 80, 120, 143]),
      ]
    

    steps = range(sum([len(steps) for _, _, _, steps in self.models]))
    self.config_evaluation = {
        'T0_zero_shot_evalset': {
            'list_dataset': self.list_zero_shot,
            'steps': steps,
            'last_train': 0
        },
        'asset': {
            'list_dataset': [('asset', 'simplification_1', 'validation')],
            'steps': steps, 
            'last_train': 4
      },
        'wiki_auto': {
            'list_dataset': [('wiki_auto', 'simplification_1', 'test')],
            'steps': steps, 
            'last_train': 4
        },
        'gigaword_constrain': {
            'list_dataset': [('gigaword', 'constrain_start+make_a_title', 'test'),
                            ('gigaword', 'constrain_contain+make_a_title', 'test'),
                            ('gigaword', 'constrain_end+make_a_title', 'test')],
            'steps': [s for s in steps if s not in range(4)], 
            'last_train': 8
        },
        'haiku': {
            'list_dataset': [('haiku', 'do_nothing', 'test')],
            'steps': [s for s in steps if s not in range(8)],
            'last_train': 12
        },
        'covid_qa_deepset': {
            'list_dataset': [('covid_qa_deepset', 'covid_cloze_book_qa', 'train')],
            'steps': [s for s in steps if s not in range(12)],
            'last_train': 16
        },
        'eli5': {
            'list_dataset': [('eli5', 'generate_a_question_1', 'test_asks')],
            'steps': [s for s in steps if s not in range(16)],
            'last_train': 20
        },
        'empathetic_dialogues': {
            'list_dataset': [('empathetic_dialogues', 'dialogue_with_emotion', 'test')],
            'steps': [s for s in steps if s not in range(20)],
            'last_train': 24
        },
        'eSNLI': {
            'list_dataset': [('eSNLI', 'explain_why', 'test')],
            'steps': [s for s in steps if s not in range(24)],
            'last_train': 28
        },
        'twitter_top20': {
            'list_dataset': [('twitter_top20', 'tweet_as+about', 'test')],
            'steps': [s for s in steps if s not in range(28)],
            'last_train': 32
        },
    }
    if self.print_detail_T0:
      for k, k_prompt, k_mode in self.list_zero_shot:
        self.config_evaluation[k] = {'list_dataset': [(k, k_prompt, k_mode)], 'steps': steps}


  def loadMapNameDict(
      self,
      d_map_metrics,
      whatMetric,
      modelNames,
      d_map_tasks
      ):

    self.whatMetric = whatMetric
    if whatMetric is None:
      self.whatMetric = self.whatMetricDefault

    self.d_map_metrics = d_map_metrics
    if d_map_metrics is None:
      self.d_map_metrics = {
          'accuracy': 'Acc',
          'start': 'Cons',
          'contain': 'Cons',
          'end': 'Cons',
          'sari': 'SARI',
          'bleu': 'B4',
          'rouge1': 'R1',
          'eq_weighted': r'$H_{cust}$',
          'jensenFirstToken': '1Tok',
          'BERTScore(f1)': 'BS',
          'CLF_acc': 'Clf',
      }

      self.d_map_tasks = d_map_tasks
      if d_map_tasks is None:
        self.d_map_tasks = {
            'T0_zero_shot_evalset': 'T0zs',
            'wiki_auto': 'Simp',
            'asset': 'ASSET',
            'gigaword_constrain': 'HGen',
            'gigaword': 'HGen',
            'haiku': 'Haiku',
            'covid_qa_deepset': 'CQA',
            'eli5': 'InqQG',
            'empathetic_dialogues': 'EmDg',
            'eSNLI': 'Exp',
            'twitter_top20': 'TwSt'
        }

      self.modelNames = modelNames
      if modelNames is None:
        self.modelNames = [
           'T0', '+Simp','+HGen',
           '+Haiku','+CQA', '+InqQG',
           '+EmDg','+Exp', '+TwSt'
        ]

  @staticmethod
  def whatMetricDefault(
    dataset_name, 
    prompt_name, 
    default_nlg=['bleu'], 
    default_nlu=['accuracy']
    ):
  
    nlg_datasets = {'haiku', 'eli5', 'wiki_auto', 'gigaword', 'covid_qa_deepset', 'empathetic_dialogues', 'twitter_top20', 'eSNLI'}
    nlu_datasets = { 'rte', 'copa', 'wic', 'winogrande', 'hellaswag', 'anli', 'cb', 'wsc', 'story_cloze', 'covidfact', 'rank_summary'}

    if 'constrain' in prompt_name:
      if 'constrain_start' in prompt_name:
        metrics = ['start']
      elif 'constrain_contain' in prompt_name:
        metrics = ['contain']
      elif 'constrain_end' in prompt_name:
        metrics = ['end']
    elif dataset_name == 'asset' or dataset_name == 'wiki_auto': 
      metrics = ['bleu', 'sari']
    elif dataset_name == 'haiku': 
      metrics = ['eq_weighted']
    elif dataset_name == 'eli5': 
      metrics = ['BERTScore(f1)'] #'jensenFirstToken', 'rouge1', 
    elif dataset_name == 'empathetic_dialogues': 
      metrics = ['BERTScore(f1)']
    elif dataset_name == 'covid_qa_deepset': 
      metrics = ['BERTScore(f1)']
    elif dataset_name == 'twitter_top20': 
      metrics = ['CLF_acc']
    elif dataset_name == 'eSNLI': 
      metrics = ['BERTScore(f1)']
    elif dataset_name in nlg_datasets: 
      metrics = default_nlg
    elif dataset_name in nlu_datasets: 
      metrics = default_nlu
    else:
      raise NotImplementedError
    
    return metrics

  def get_key(
        self,
        dataset_name, eval_mode, prompt_name, 
        model_name, model_from, model_rehearsal, 
        step
      ):
      
      if 'T0' in model_name:
        model_name = 'T0_3B' if self.model_size == "3B" else 'T0pp'
        key = f'{model_name}.{dataset_name}.{eval_mode}.{prompt_name}'
      else:
        key = f'{model_name}.{self.model_size}.rehearsal{model_rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'
        if model_from != None:
          # to fixe, different formating name 3B Vs 11B
          if self.model_size == '3B':
            key = key + f'.{"->".join(model_from)}'
      
      return key

  def getScoreTable(self, model, eval_modes):
      
      model_name, model_from, model_rehearsal, model_steps = model

      d_res = {
          self.d_map_metrics[m]: [] 
          for m 
          in self.whatMetric(eval_modes[0][0], eval_modes[0][1]) 
          #in self.whatMetric(eval_mode[0], eval_mode[1]) for eval_mode in eval_modes
      }

      for dataset_name, prompt_name, eval_mode in eval_modes:
        key = self.get_key(
            dataset_name, eval_mode, prompt_name, 
            model_name, model_from, model_rehearsal, 
            model_steps[-1], 
        )
        
        for m in self.whatMetric(dataset_name, prompt_name):
          try:
            score = self.d_scores[key][m]
            if m in {'accuracy', 'bleu', 'rouge1', 'CLF_acc', 'contain', 'end', 'start', 'eq_weighted', 'BERTScore(f1)'}:
              score *= 100
            d_res[self.d_map_metrics[m]].append(score)
          except:
            print(key, m)           


      for m in list(d_res.keys()):
        d_res[m] = np.average(d_res[m])

      return d_res

  def d_res2df(self, d_res):
      
      def do_format(d):
        return '/'.join([str(round(v, 1)) for k, v in d.items()])

      df = pd.DataFrame(d_res)
      for col in df.columns:
        df[col] = df[col].apply(do_format)

      types_header_for_insert = ['/'.join(list(k.keys())) for k in pd.DataFrame(d_res).loc[0]]
      df.columns = pd.MultiIndex.from_tuples(zip([self.d_map_tasks[c] for c in df.columns], types_header_for_insert))

      df = df.rename(index={i: k for i, k in enumerate(self.modelNames)
      })

      return df

  def getTable(self):
    d_res = {}
    for eval_name, eval_modes in self.config_evaluation.items():
      
      results = []
      for model in self.models:    
        results.append(self.getScoreTable(model, eval_modes['list_dataset']))
      d_res[eval_name] = results

    return self.d_res2df(d_res)


  def getScoresFigure(
    self,
    default_nlg=['bleu'], default_nlu=['accuracy']
    ):

    scores = {}

    for group_name, group_datasets in self.config_evaluation.items(): 

      last_step = 0
      all_steps = []
      scores[group_name] = []
      
      for model_name, model_from, rehearsal, steps in self.models:
        for step in steps:
          all_steps.append(last_step + step)

          step_score = 0
          for dataset_name, prompt_name, eval_mode in group_datasets['list_dataset']:
            # calculate the score

            key = self.get_key(dataset_name, eval_mode, prompt_name, model_name, model_from, rehearsal, step)
            
            nlg_metric = default_nlg if dataset_name != 'wiki_auto' else 'sari'
            if key not in self.d_scores:
              print(f'Break for {group_datasets["list_dataset"]}: {key} does not exist')
              step_score = None
              break
            metrics = self.whatMetric(dataset_name, prompt_name, nlg_metric, default_nlu)
            step_score += np.average([self.d_scores[key][m] for m in metrics])
          
          if step_score is not None:
              step_score = step_score/len(group_datasets['list_dataset'])
          scores[group_name].append(step_score)
        
        last_step += step
      
    return scores, all_steps

  def printSequencialFigure(
      self,
      save_dir, 
      do_normalise=True, 
      default_nlg=['bleu'], default_nlu=['accuracy'],
      ):
  

    scores, all_steps = self.getScoresFigure(
        default_nlg=default_nlg, 
        default_nlu=default_nlu, 
    )
  
    
    plt.figure(figsize=(8, 6))
    plt.xlabel('Steps')
    plt.ylabel('Relative Gain')

    for k_name, v_scores in scores.items():  

      if do_normalise:
          #v_scores = [v_scores[i]/v_scores[0]-1 for i in range(len(v_scores))]
          v_scores = [v_scores[i]/v_scores[self.config_evaluation[k_name]['last_train']] 
                      if v_scores[i] else None  for i in range(len(v_scores))]


      x = self.config_evaluation[k_name]['steps']
      y = [v_scores[i] for i in self.config_evaluation[k_name]['steps']]

      plt.plot(x, y, label=self.d_map_tasks[k_name])

    plt.xticks(range(len(all_steps)), all_steps, rotation='vertical')
    plt.legend() #bbox_to_anchor=(1.1, 1.05)
    if save_dir is not None:
      plt.savefig(os.path.join(save_dir, self.model_size + '.' + '->'.join(self.models[-1][1] + [self.models[-1][0]])), format='pdf', bbox_inches='tight')
    plt.show()
    
    return scores

  def printSequencialFigureV2(
      self,
      save_dir, 
      do_normalise=True, 
      default_nlg=['bleu'], default_nlu=['accuracy'], 
    ):
  

    scores, all_steps = self.getScoresFigure(
        default_nlg=default_nlg, 
        default_nlu=default_nlu, 
    )
    
    
    nb_points_per_model = 5
    score_trained = [[] for _ in all_steps]
    
    plt.figure(figsize=(8, 6))
    plt.xlabel('Steps')
    plt.ylabel('Relative Gain')

    labels = []
    for k_name, v_scores in scores.items():  

      if do_normalise:
          #v_scores = [v_scores[i]/v_scores[0]-1 for i in range(len(v_scores))]
          v_scores = [v_scores[i]/v_scores[self.config_evaluation[k_name]['last_train']] 
                      if v_scores[i] else None  for i in range(len(v_scores))]


      x = self.config_evaluation[k_name]['steps']
      y = [v_scores[i] for i in self.config_evaluation[k_name]['steps']]


      
      if k_name != 'T0_zero_shot_evalset':
        
        for idx in x[nb_points_per_model:]:
          score_trained[idx].append(v_scores[idx])

        x = x[:nb_points_per_model]
        y = y[:nb_points_per_model]
        linestyle = (0, (3, 9))
        color = None
      else:
        linestyle = 'solid'
        color = 'g'
      plt.plot(x, y, label=self.d_map_tasks[k_name], linestyle=linestyle, color=color)


    # plot score_trained:
    x = range(len(all_steps))[nb_points_per_model:]
    y = [np.average(scores) for scores in score_trained][nb_points_per_model:]
    plt.plot(x, y, label='TrainedTasks', color='b')
    plt.xticks(range(len(all_steps)), all_steps, rotation='vertical')
    plt.legend() #bbox_to_anchor=(1.1, 1.05)
    if save_dir is not None:
      plt.savefig(os.path.join(save_dir, self.model_size + '.' + '->'.join(models[-1][1] + [models[-1][0]]) + '.v2'), format='pdf', bbox_inches='tight')

    plt.show()

    return scores
    
  def getFigureSequential(self, save_dir, version='V1', do_normalise=True):

    if version == 'V1':
      printSequencialFigure = self.printSequencialFigure
    else:
      printSequencialFigure = self.printSequencialFigureV2

    fig_scores = printSequencialFigure(
      save_dir = save_dir, 
      do_normalise=do_normalise,
    ) 

    return fig_scores

  def getFigureNonSequential(
    self,
    save_dir,
    nlg_metric=['bleu'], 
    nlu_metric=['accuracy'],
    do_normalise=True,
    get_color_custom=None,
    d_line_styles=None,
    ):
  
  
    def getColorNonSeq(group_name):

      color = None
      if group_name == 'T0_zero_shot_evalset':
        color = 'lime'
      elif group_name in {'wiki_auto', 'haiku', 'gigaword_constrain'} :
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

    if not get_color_custom:
      get_color_custom = getColorNonSeq
    
    if d_line_styles is None:
      d_line_styles = {0: (0, (3, 9)), 250: (0, (9, 3)), 1000: 'solid'}
    

    def getFig(model_name, d_datasets):

      d_scores_fig = {}
      for group_name, group_datasets in d_datasets.items():
        for rehearsal, steps in self.non_seq_d_rehearsals.items():
            scores = []
            for step in steps:
              step_scores = []
              for dataset_name, prompt_name, eval_mode in group_datasets:

                if step == 0:
                  baseline_name = 'T0_3B' if self.model_size == '3B' else 'T0pp'
                  key = f'{baseline_name}.{dataset_name}.{eval_mode}.{prompt_name}'
                else:
                  key = f'{model_name}.{self.model_size}.rehearsal{rehearsal}.{step}.{dataset_name}.{eval_mode}.{prompt_name}'
                    
                metrics = self.whatMetric(dataset_name, prompt_name, nlg_metric, nlu_metric)
                step_scores.append(np.average([self.d_scores[key][m] for m in metrics]))
                
              scores.append(sum(step_scores)/len(step_scores))
            
            d_scores_fig[f'{group_name}.{rehearsal}'] = scores
            if do_normalise:
              scores = [scores[i]/scores[0]-1 for i in range(len(steps))]
            
            plt.plot(scores, label=f'{self.d_map_tasks[group_name]}({rehearsal/1000}%)', color=get_color_custom(group_name), linestyle=d_line_styles[rehearsal])

      d_mini_map = {
          'gigaword': 'Constrain (Cons)',
          'wiki_auto': 'Simplification (Simp)',
          'haiku': 'Haiku'
      }
      plt.xlabel('Steps')
      plt.ylabel('Relative Gain', fontsize=15)
      plt.xticks(range(len(steps)), steps, fontsize=15)
      plt.legend(prop={'size': 11}, loc='lower right') #bbox_to_anchor=(1.1, 1.05)
      plt.title(d_mini_map[model_name], fontsize=20)
      plt.savefig(os.path.join(save_dir, self.model_size + '.' + f'{model_name}.{"normalized" if do_normalise else "absolute"}'), format='pdf')
      plt.show()
    
    for model_name, d_datasets in self.non_seq_config_images.items():
      getFig(model_name, d_datasets)

    return 

