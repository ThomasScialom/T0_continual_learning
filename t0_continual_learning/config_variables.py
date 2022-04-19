t0_train_datasets = ['commonsense_qa', 'dream', 'quail', "quartz", 'social_i_qa', 'wiqa', 'cosmos_qa', 'qasc', 'quarel', 'sciq', 'wiki_hop', 
'adversarial_qa_dbert', 'adversarial_qa_dbidaf', 'adversarial_qa_droberta', 'quoref', 'duorc_ParaphraseRC', 'duorc_SelfRC', 'ropes', 'wiki_qa', 'common_gen', 'wiki_bio',
'app_reviews', 'amazon_polarity', 'imdb', 'rotten_tomatoes', 'gigaword', 'cnn_dailymail', 'multi_news', 'samsum',
'xsum', 'ag_news', 'dbpedia_14', 'trec', 'paws_labeled_final', 'glue_mrpc', 'glue_qqp', 'yelp_review_full', 'kilt_tasks_hotpotqa']


# ___Usage: for function **process_datasets** to load and format automatically the datasets___

continual_train = {
  'gigaword': {
      'train': [
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
      ],
  },
  'wiki_auto': {
      'train': [
        ('custom', 'simplification_1'),
      ]
  },
  'eli5': {
      'train_asks': [
        ('custom', 'generate_a_question_2'),
      ]
  },
  'haiku': {
      'train': [
        ('custom', "do_nothing"),
      ]
  },
  'covid_qa_deepset': {
      'train': [
        ('custom', "covid_cloze_book_qa"),
      ]
  },
  'empathetic_dialogues': {
      'train': [
        ('custom', "dialogue_with_emotion"),
      ]
  },
  'twitter_top20': {
      'train': [
        ('custom', "tweet_as+about"),
      ]
  },
  'eSNLI': {
      'train': [
        ('custom', "explain_why"),
      ]
  },
}

continual_test = {
  'gigaword': {
      'test': [
        ('t0_template', 'make_a_title'),
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
        ('t0_template', 'write_its_sentence'),
        ('custom', 'constrain_start+write_its_sentence'),
        ('custom', 'constrain_contain+write_its_sentence'),
        ('custom', 'constrain_end+write_its_sentence'),
      ]
  },
  'wiki_auto': {
      'test': [
       ('custom', 'simplification_1'),
       ('custom', 'simplification_2'),
      ] 
  },
  'asset': {
      'validation': [
        ('custom', 'simplification_1'),
        ('custom', 'simplification_2')
      ]
  },
  'eli5': {
      'test_asks': [
        ('custom', 'generate_a_question_1'),
      ]
  },
  'haiku': {
      'test': [
        ('custom', "do_nothing"),
      ]
  },
  'covid_qa_deepset': {
      'train': [
        ('custom', "covid_cloze_book_qa"),
      ]
  },
  'empathetic_dialogues': {
      'test': [
        ('custom', "dialogue_with_emotion"),
      ]
  },
  'twitter_top20': {
      'test': [
        ('custom', "tweet_as+about"),
      ]
  },
  'eSNLI': {
      'test': [
        ('custom', "explain_why"),
      ]
  },
  'winogrande': { 'validation': [('t0_template', "fill in the blank")]},
  'rte': {'validation': [('t0_template', "can we infer")]},
  'wic': {'validation': [('t0_template', "same_sense")]},
  'copa': {'validation': [('t0_template', "choose")]}, 
  'hellaswag': {'validation': [('t0_template', "__RANDOM__")]}, 
  'anli': {'test_r1': [('t0_template', "__RANDOM__")]}, 
  'cb': {'validation': [('t0_template', "__RANDOM__")]}, 
  'wsc': {'validation': [('t0_template', "__RANDOM__")]}, 
  'story_cloze': {'validation': [('t0_template', "__RANDOM__")]}, 
}

# ____________________________________________________________________________________________
# ___Usage: for function **format2train** to format the training files with the correct number of rehearsal examples

d_config_reharsal_non_seq = {
  "gigaword": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'gigaword',
          'prompts': {
              'constrain_start+make_a_title': 33000,
              'constrain_contain+make_a_title': 33000,
              'constrain_end+make_a_title': 34000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
  'wiki_auto': {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'wiki_auto',
          'prompts': {
              'simplification_1': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
  'eli5': {
      'new_dataset': {
          'eval_mode': 'train_asks',
          'name': 'eli5',
          'prompts': {
              'generate_a_question_1': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
  'haiku': {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'haiku',
          'prompts': {
              'do_nothing': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
  'covid_qa_deepset': {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'covid_qa_deepset',
          'prompts': {
              'covid_cloze_book_qa': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
}

d_config_reharsal_seq = {
  'wiki_auto': {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'wiki_auto',
          'prompts': {
              'simplification_1': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': None
      }
  },
  "sequencial.gigaword.from.wiki_auto": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'gigaword',
          'prompts': {
              'constrain_start+make_a_title': 33000,
              'constrain_contain+make_a_title': 33000,
              'constrain_end+make_a_title': 34000,
          }
      },
      'reharsal': {'inheritFrom': 'wiki_auto'}
  },
  "sequencial.haiku.from.wiki_auto->gigaword":  {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'haiku',
          'prompts': {
              "do_nothing": 100000,
          }
      },
      'reharsal': {'inheritFrom': 'sequencial.gigaword.from.wiki_auto'}
  },
  "sequencial.covid_qa_deepset.from.wiki_auto->gigaword->haiku": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'covid_qa_deepset',
          'prompts': {
              'covid_cloze_book_qa': 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.haiku.from.wiki_auto->gigaword"}
  },
  "sequencial.eli5.from.wiki_auto->gigaword->haiku->covid_qa_deepset": {
      'new_dataset': {
          'eval_mode': 'train_asks',
          'name': 'eli5',
          'prompts': {
              "generate_a_question_2": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.covid_qa_deepset.from.wiki_auto->gigaword->haiku"}
  },
  
  "sequencial.empathetic_dialogues.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'empathetic_dialogues',
          'prompts': {
              "dialogue_with_emotion": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.eli5.from.wiki_auto->gigaword->haiku->covid_qa_deepset"}
  },
  "sequencial.eSNLI.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5->empathetic_dialogues": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'eSNLI',
          'prompts': {
              "explain_why": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.empathetic_dialogues.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5"}
  },
  "sequencial.twitter_top20.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5->empathetic_dialogues->eSNLI": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'twitter_top20',
          'prompts': {
              "tweet_as+about": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.eSNLI.from.wiki_auto->gigaword->haiku->covid_qa_deepset->eli5->empathetic_dialogues"}
  },
}

d_config_reharsal_seq_inverse = {
  "twitter_top20": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'twitter_top20',
          'prompts': {
              "tweet_as+about": 100000,
          }
      },
      'reharsal': {'inheritFrom': None}
  },
  "sequencial.eSNLI.from.twitter_top20": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'eSNLI',
          'prompts': {
              "explain_why": 100000,
          }
      },
      'reharsal': {'inheritFrom': "twitter_top20"}
  },
  "sequencial.empathetic_dialogues.from.twitter_top20->eSNLI": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'empathetic_dialogues',
          'prompts': {
              "dialogue_with_emotion": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.eSNLI.from.twitter_top20"}
  },
 "sequencial.eli5.from.twitter_top20->eSNLI->empathetic_dialogues": {
      'new_dataset': {
          'eval_mode': 'train_asks',
          'name': 'eli5',
          'prompts': {
              "generate_a_question_2": 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.empathetic_dialogues.from.twitter_top20->eSNLI"}
  },
 "sequencial.covid_qa_deepset.from.twitter_top20->eSNLI->empathetic_dialogues->eli5": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'covid_qa_deepset',
          'prompts': {
              'covid_cloze_book_qa': 100000,
          }
      },
      'reharsal': {'inheritFrom': "sequencial.eli5.from.twitter_top20->eSNLI->empathetic_dialogues"}
  },
 "sequencial.haiku.from.twitter_top20->eSNLI->empathetic_dialogues->eli5->covid_qa_deepset":  {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'haiku',
          'prompts': {
              "do_nothing": 100000,
          }
      },
      'reharsal': {'inheritFrom': 'sequencial.covid_qa_deepset.from.twitter_top20->eSNLI->empathetic_dialogues->eli5'}
  },
 "sequencial.gigaword.from.twitter_top20->eSNLI->empathetic_dialogues->eli5->covid_qa_deepset->haiku": {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'gigaword',
          'prompts': {
              'constrain_start+make_a_title': 33000,
              'constrain_contain+make_a_title': 33000,
              'constrain_end+make_a_title': 34000,
          }
      },
      'reharsal': {'inheritFrom': 'sequencial.haiku.from.twitter_top20->eSNLI->empathetic_dialogues->eli5->covid_qa_deepset'}
  },
  'sequencial.wiki_auto.from.twitter_top20->eSNLI->empathetic_dialogues->eli5->covid_qa_deepset->haiku->gigaword': {
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'wiki_auto',
          'prompts': {
              'simplification_1': 100000,
          }
      },
      'reharsal': {
          'inheritFrom': 'sequencial.gigaword.from.twitter_top20->eSNLI->empathetic_dialogues->eli5->covid_qa_deepset->haiku'
      }
  },
}

# ____________________________________________________________________________________________
# ___Usage: different evaluation sets group per type to run predictions systematically on them

evaluation_T0evalsets = {
  'T0_train_eval':{
    'test':{
      'type': 't0_template',
      'choice': '',
      'metrics': ['bleu']
    }
  },
  'winogrande': { 
      'validation': {
          "fill in the blank": {
            'type': 't0_template',
            'choice': 'option1_option2',
            'metrics': ['accuracy']
          }
      }
  },
  'rte': { 
      'validation': {
          "can we infer": {
            'type': 't0_template',
            'choice': 'yes_no',
            'metrics': ['accuracy']
          }
      }
  },
  'wic': { 
      'validation': {
          "same_sense": {
            'type': 't0_template',
            'choice': 'yes_no',
            'metrics': ['accuracy']
          }
      }
  },
  'copa': { 
    'validation': {
      "choose": {
        'type': 't0_template',
        'choice': 'option1_option2',
        'metrics': ['accuracy']
      }
    }
  },
  'hellaswag': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['accuracy']
          }
      }
  },
  'anli': { 
      'test_r1': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['accuracy']
          }
      }
  },
  'cb': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['accuracy']
          }
      }
  },
  'wsc': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['accuracy']
          }
      }
  },

  'story_cloze': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['accuracy']
          }
      }
  },
}




evaluation_new_tasks = {
  'gigaword': {
      'test': {
        'make_a_title': {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu']
        },
        'constrain_start+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_start']
        },
        'constrain_contain+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_contain']
        },
        'constrain_end+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_end']
        },
      },
  },
  'wiki_auto': {
      'test': {
        'simplification_1': {
            'type': 'custom',
            'choice': '',
            'metrics': ['bleu', 'sari']
        }
      }
  },
  'asset': {
    'validation': {
      'simplification_1': {
        'choice': '',
        'metrics': ['bleu', 'sari'],
        'type': 'custom'
      }
    }
  },
  'eli5': {
      'test_asks': {
        'generate_a_question_1': {
            'type': 'custom',
            'choice': '',
            'metrics': ['bertscore', 'rouge', 'bleu', 'firstWordSim']
        }
      }
  },
  'haiku': {
      'test': {
        'do_nothing': {
            'type': 'custom',
            'choice': '',
            'metrics': ['haikuMetric']
        }
      }
  },
  'covid_qa_deepset': {
      'train': {
        'covid_cloze_book_qa': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'bertscore']
        }
      }
  },
  'empathetic_dialogues': {
      'test': {
        'dialogue_with_emotion': {
            'type': 'custom',
            'choice': '',
            'metrics': ['bleu', 'bertscore']
        }
      }
  },
  'eSNLI': {
      'test': {
        'explain_why': {
            'type': 'custom',
            'choice': '',
            'metrics': ['bertscore', 'bleu']
        }
      }
  },
  'twitter_top20': {
      'test': {
        'tweet_as+about': {
            'type': 'custom',
            'choice': '',
            'metrics': ['bleu', 'CLF_acc', 'bertscore']
        }
      }
  },
}



evaluation_zero_shot  = {
  'gigaword': {
      'test': {
        'write_its_sentence': {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge']
        },
        'constrain_start+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
        'constrain_contain+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
        'constrain_end+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
      }
  },
  'wiki_auto': {
      'test': {
        'simplification_2': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', ]
        }
      },
  },
}
