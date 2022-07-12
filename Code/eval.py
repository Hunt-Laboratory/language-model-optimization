
## Imports.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
from soft_embedding import SoftEmbedding
from typing import Dict

import json
import pandas as pd
import dill as pickle

path = ""
ops = pickle.load( open( path + 'data/ops.pickle', "rb" ) )


class ArgOpDataset(torch.utils.data.Dataset):
    """Argument operation dataset."""

    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
    
        item = {key: val[idx] for key, val in self.texts.items()}

        item["labels"] = torch.sub(
            item["input_ids"] * item["attention_mask"],
            100 * torch.sub(1, item["attention_mask"])
        )
        return item

    def __len__(self):
        return len(self.texts['input_ids'])


def build_dataset(CONFIG, split, tokenizer):

    CF = CONFIG
    op = CF['op']
    prompt_type = CF['prompt_type']

    # Retrieve data.
    data = ops[op][split]

    if split == 'train':
        data = data[:10]
    
    # Construct prompts.
    prompts = [prompt[prompt_type]['in'] + prompt[prompt_type]['out'] for prompt in data]

    prompts = [p for p in prompts if len(p) < 2500]

    # Tokenize.
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    # print(inputs)

    # If using soft prompts, pad inputs to correct length.
    if bool(CF['strategy_softprompt']):
        inputs['input_ids'] = torch.cat([torch.full((inputs['input_ids'].size()[0],30), 50256, dtype=torch.long), inputs['input_ids']], 1)
        inputs['attention_mask'] = torch.cat([torch.full((inputs['input_ids'].size()[0],30), 1, dtype=torch.long), inputs['attention_mask']], 1)

    prompt_lengths = [len(p) for p in tokenizer([prompt[prompt_type]['in'] for prompt in data])['input_ids']]
    for prompt_idx in range(len(prompts)):
        prompt_length = prompt_lengths[prompt_idx]
        if bool(CF['strategy_softprompt']):
            prompt_length = prompt_length + 30
        prompt_length = min(prompt_length, list(inputs['input_ids'][prompt_idx].size())[0])
        prompt_lengths[prompt_idx] = prompt_length
    inputs['prompt_lengths'] = prompt_lengths

    return ArgOpDataset(inputs)


class MyTrainer(Trainer):

    def __init__(self, *args, config={}, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.config = config
        self.metrics = []


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Mask prompt labels so they are not included in perplexity
        inputs['labels'] = inputs['input_ids'].clone().detach()
        for prompt_idx in range(len(inputs['prompt_lengths'])):
            prompt_length = inputs['prompt_lengths'][prompt_idx]
            for token_idx in range(prompt_length):
                inputs['labels'][prompt_idx][token_idx] = -100
        del inputs['prompt_lengths']

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        CF = self.config

        # Read in metrics, if resuming from previous.
        metrics_filepath = f"{path}outputs/metrics/n{CF['id']}.csv"
        
        print(output)

        # Add current log.
        if 'eval_loss' in output:
            self.metrics.append(['Test', 'Perplexity', output['eval_loss']])

        # Write to CSV.
        t_metrics = [row for row in self.metrics if row[2] <= 100]
        metrics = pd.DataFrame(self.metrics, columns=['Split', 'Metric', 'Value'])
        metrics.to_csv(metrics_filepath, index=False)



def optimise_model(CF):

    torch.manual_seed(5678)

    checkpoint = CF['foundation']

    # Prepare datasets.

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '[PAD]'

    train_set = build_dataset(CF, 'train', tokenizer)
    val_set = build_dataset(CF, 'test', tokenizer)
    
    # Prepare model.

    print('LOADING FOUNDATION MODEL...')
    net = GPTNeoForCausalLM.from_pretrained(checkpoint)

    print('LOADING FINEDTUNED WEIGHTS...')
    # Load pre-trained checkpoints (where relevant).
    if bool(CF['strategy_softprompt']):
        s_wte = SoftEmbedding(net.get_input_embeddings(), n_tokens=30, initialize_from_vocab=True)
        net.set_input_embeddings(s_wte)
        net.load_state_dict(torch.load(f"{CF['checkpoint']}/pytorch_model.bin"), strict=False)
    elif CF['checkpoint'] != 'EleutherAI/gpt-neo-2.7B': 
        net.load_state_dict(torch.load(f"{CF['checkpoint']}/pytorch_model.bin"), strict=False)
    
    # Freeze all parameters so no actual training occurs.
    for name, param in net.named_parameters():
        param.requires_grad = False

    # Define training args.
    training_args = TrainingArguments(
        output_dir = f"outputs/checkpoints/{CF['id']}",
        seed = 5678,
        logging_strategy = "steps",
        logging_steps = 1,
        logging_first_step = True,
        evaluation_strategy = "steps",
        eval_steps = 1,
        save_strategy = "no",
        ddp_find_unused_parameters = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 32,
        num_train_epochs = 1,
        max_steps = 1,
        ignore_data_skip = False
    )

    trainer = MyTrainer(
        model = net,
        config = CF,
        args = training_args,
        train_dataset = train_set,
        eval_dataset = val_set
    )

    trainer.evaluate()

    # Clear memory.

    del net
    del trainer
    torch.cuda.empty_cache()




## Optimise Models

with open( path + 'config/eCONFIG.json', "r") as f:
    CONFIG = json.load(f)

for idx, CF in enumerate(CONFIG):
    if not CF['complete']:
        optimise_model(CF)
        if not CF['dry_run']:
            CONFIG[idx]['complete'] = 1

with open( path + 'config/eCONFIG.json', 'w') as f:
    json.dump(CONFIG, f)