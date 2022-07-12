
This folder contains the scripts used to train, evaluate, and generate completions for prompts describing argumentative NLP tasks from optimised versions of GPT-Neo-2.7B. They were created as part of the AI4DM Phase 2 project by Luke Thorburn at the Hunt Lab for intelligence research at the University of Melbourne. For long term support, please use t.gelder@unimelb.vic.edu.au as the point of contact.

The main scripts are:

- `data.ipynb` - This is the main Jupyter notebook used for generating the training, validation and test sets used for model training and evaluation. You should not need to run it because the datasets are available to you, precompiled, in `data/ops.pickle` and `data/ops_raw.pickle`. But it is included for reference, because it contains the logic used to clean the data and the wording of the prompt templates used.

- `train.py` - This is the main script used for model training. It is set up to be run using DeepSpeed, with the command `deepspeed train.py --deepspeed`. It requires substantial GPU capacity, likely with 2 or more GPUs, each with at least 20GB of vRAM. Error messages should be informative regarding missing dependences, but make sure that your versions of DeepSpeed and PyTorch are compatible, and compatible with the versions of CUDA and your GPU drivers. This was the biggest sticking point for me.

- `eval.py` - This is the main script used for calculating perplexity scores on the test set. It should be able to be run on a modestly powered machine with `python eval.py`, but you will need to have pre-downloaded the pretrained models and positioned them in the appropriate folders relative to the script.

- `gen.ipynb` - This is the main Jupyter notebook used for generating completed responses for evaluation purposes using the pretrained models. It should be able to be run on a modestly powered machine, but you will need to have pre-downloaded the pretrained models and positioned them in the appropriate folders relative to the script.