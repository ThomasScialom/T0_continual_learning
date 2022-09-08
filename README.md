# T0_continual_learning
Adding new tasks to LLM without catastrophic forgetting 

### Paper: 
https://arxiv.org/abs/2205.12393

### Models: [Todo]
https://huggingface.co/ThomasNLG/ct0

### Code:
We have not clean all the code yet, but most of the different steps can be found in this [Collab](https://colab.research.google.com/drive/1Wp2mk5Dbzw5PAGcMOuuE_xB-6gFV0800#scrollTo=AIGI4ahyrD2s). 

In particular the notebook contains 
* The steps to create the dataset and format them with reheasarl. Note that for some raw datasets it might not be possible to still download them (e.g. broken link), but you can find them in our [main folders](https://drive.google.com/drive/folders/1aQmnPmYGoQIYPK5jgbv4K4PXYYNwqisH?usp=sharing), just as the formated datasets. 
* The evaluation scripts.

For training, we plan to release the scripts but nothing fancy, simply finetuning T5 with HF framework, with the parameters mentioned in our paper. 

### Material:
All the material used and created or used in the notebook etc., including the training data, the predictions and the checkpoints are publicly available in [main folders](https://drive.google.com/drive/folders/1aQmnPmYGoQIYPK5jgbv4K4PXYYNwqisH?usp=sharing)
