
This folder contains the scripts used to train, evaluate, and generate completions for prompts describing argumentative NLP tasks from optimised versions of GPT-Neo-2.7B.

The main scripts are:

- `data.ipynb` - This is the main Jupyter notebook used for generating the training, validation and test sets used for model training and evaluation. We are not able to make the datasets public due to constraints outside our control, but we include the code as a reference for what data cleaning operations were performed, as well as the prompt templates that were used. Contact us if you need access to the dataset and we will do our best to help.

- `train.py` - This is the main script used for model training. It is set up to be run using DeepSpeed, with the command `deepspeed train.py --deepspeed`. It requires substantial GPU capacity, likely with 2 or more GPUs, each with at least 20GB of vRAM. Error messages should be informative regarding missing dependences, but make sure that your versions of DeepSpeed and PyTorch are compatible, and compatible with the versions of CUDA and your GPU drivers. This was the biggest sticking point for me. The configuration for the models which this script trains are specified in `config/CONFIG.json`.

- `eval.py` - This is the main script used for calculating perplexity scores on the test set. It should be able to be run on a modestly powered machine with `python eval.py`, but you will need to have pre-downloaded the pretrained models and positioned them in the appropriate folders relative to the script. The specification that determines which models are evaluated is in `config/eCONFIG.json`.

- `gen.ipynb` - This is the main Jupyter notebook used for generating completed responses for manual evaluation purposes using the pretrained models. It should be able to be run on a modestly powered machine, but you will need to have pre-downloaded the pretrained models and positioned them in the appropriate folders relative to the script. The specification that determined from which models to generate is included in `config/gCONFIG.json`. More generally, it can be used as a model for how to generate new completions of arguments using the pretrained models available on Huggingface.