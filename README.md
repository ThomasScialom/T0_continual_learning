# T0_continual_learning
Adding new tasks to LLM without catastrophic forgetting 

### Paper: 
https://arxiv.org/abs/2205.12393

### Models:
https://huggingface.co/ThomasNLG/CT0-11B

### Code:
We haven't cleaned up all the code yet, but most of the different steps can be found in this [Collab](https://colab.research.google.com/drive/1Wp2mk5Dbzw5PAGcMOuuE_xB-6gFV0800#scrollTo=AIGI4ahyrD2s).

In particular the notebook contains 
* The steps to create the dataset and to format them with the reheasarl. Note that for some raw datasets it might not be possible anymore to download them (e.g. broken link on HF hub). You can still find the processed data in our [main folders](https://drive.google.com/drive/folders/1aQmnPmYGoQIYPK5jgbv4K4PXYYNwqisH?usp=sharing), just as the formated datasets. 
* The evaluation scripts.

For training, we plan to release the scripts. But you dont wait for it, we applied nothing fancy, simply finetuning T5 using the standard HF framework. All tehe parameters are mentioned in our paper. 

### Material:
All the material required in the notebook etc., including the training data, the predictions and the checkpoints are publicly available in [main folders](https://drive.google.com/drive/folders/1aQmnPmYGoQIYPK5jgbv4K4PXYYNwqisH?usp=sharing)
