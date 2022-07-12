
## Imports.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from soft_embedding import SoftEmbedding
from typing import Dict

import json
import pandas as pd
import altair as alt
import dill as pickle

path = ""
ops = pickle.load( open( path + 'data/ops.pickle', "rb" ) )


class ArgOpDataset(torch.utils.data.Dataset):
    """Argument operation dataset."""

    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        # print(self.texts)
        # item = tokenizer(self.texts, padding=True, truncation=True, return_tensors="pt")

        item = {key: val[idx] for key, val in self.texts.items()}

        # print(item)
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
    truncate = CF['truncate']

    # Retrieve data.
    data = ops[op][split]
    if truncate:
        data = data[:truncate]
    
    # Construct prompts.
    prompts = [prompt[prompt_type]['in'] + prompt[prompt_type]['out'] for prompt in data]

    prompts = [p for p in prompts if len(p) < 2500]

    # Tokenize.
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    # If using soft prompts, pad inputs to correct length.
    if bool(CF['strategy_softprompt']):
        inputs['input_ids'] = torch.cat([torch.full((inputs['input_ids'].size()[0],CF['soft_prompt_length']), 50256, dtype=torch.long), inputs['input_ids']], 1)
        inputs['attention_mask'] = torch.cat([torch.full((inputs['input_ids'].size()[0],CF['soft_prompt_length']), 1, dtype=torch.long), inputs['attention_mask']], 1)

    return ArgOpDataset(inputs)


class MyTrainer(Trainer):

    def __init__(self, *args, config={}, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.config = config
        self.metrics = []


    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        CF = self.config

        # Read in metrics, if resuming from previous.
        metrics_filepath = f"{path}outputs/metrics/n{CF['id']}.csv"
        if CF['resume_from_checkpoint']:
            self.metrics = pd.read_csv(metrics_filepath).values.tolist()
        
        # Add current log.
        if 'loss' in output:
            self.metrics.append([self.state.epoch, 'Train', 'Perplexity', output['loss']])
        elif 'eval_loss' in output:
            self.metrics.append([self.state.epoch, 'Validate', 'Perplexity', output['eval_loss']])

        # Write to CSV.
        t_metrics = [row for row in self.metrics if row[3] <= 100]
        metrics = pd.DataFrame(self.metrics, columns=['Epoch', 'Split', 'Metric', 'Value'])
        metrics.to_csv(metrics_filepath, index=False)
        
        # Plot.
        
        t_metrics = pd.DataFrame(t_metrics, columns=['Epoch', 'Split', 'Metric', 'Value'])
        chart = alt.Chart(t_metrics, title=f"Net {CF['id']} Perplexity").mark_line(point=True).encode(
            x='Epoch',
            y=alt.Y('Value',
                    scale=alt.Scale(domain=(0, t_metrics['Value'].max()))
            ),
            color='Split',
            strokeDash='Split'
        )
        chart.save(f"{path}outputs/graphs/n{CF['id']}.html")



def optimise_model(CF):

    torch.manual_seed(5678)

    checkpoint = CF['foundation']

    # Prepare datasets.

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '[PAD]'

    train_set = build_dataset(CF, 'train', tokenizer)
    val_set = build_dataset(CF, 'validate', tokenizer)
    
    # Prepare model.

    net = AutoModelForCausalLM.from_pretrained(checkpoint)

    # If soft-prompt enabled, insert prompt embeddings into model.
    if bool(CF['strategy_softprompt']):
        s_wte = SoftEmbedding(net.get_input_embeddings(),
                            n_tokens=CF['soft_prompt_length'], 
                            initialize_from_vocab=True)
        net.set_input_embeddings(s_wte)

    # Freeze subset of parameters (if specified).
    for name, param in net.named_parameters():
        if name == 'transformer.wte.learned_embedding':
            param.requires_grad = bool(CF['strategy_softprompt'])
        elif '.bias' in name:
            param.requires_grad = bool(CF['strategy_biases'])
        else:
            param.requires_grad = bool(CF['strategy_weights'])

    # Define training args.
    training_args = TrainingArguments(
        output_dir = f"outputs/checkpoints/{CF['id']}",
        seed = 5678,
        logging_strategy = "steps",
        logging_steps = 50,
        logging_first_step = True,
        evaluation_strategy = "steps",
        eval_steps = 50,
        save_strategy = "steps",
        save_steps = 50,
        ddp_find_unused_parameters = False,
        deepspeed = "config/ds_config.json",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = CF['batch_size'],
        # num_train_epochs = CF['n_epochs'],
        num_train_epochs = 0,
        ignore_data_skip = False
    )

    trainer = MyTrainer(
        model = net,
        config = CF,
        args = training_args,
        train_dataset = train_set,
        eval_dataset = val_set
    )

    trainer.train(
        resume_from_checkpoint=bool(CF['resume_from_checkpoint'])
    )

    # Clear memory.

    del net
    del trainer
    torch.cuda.empty_cache()




## Optimise Models

with open( path + 'config/CONFIG.json', "r") as f:
    CONFIG = json.load(f)

for idx, CF in enumerate(CONFIG):
    if not CF['complete']:
        optimise_model(CF)
        if not CF['dry_run']:
            CONFIG[idx]['complete'] = 1

with open( path + 'config/CONFIG.json', 'w') as f:
    json.dump(CONFIG, f)