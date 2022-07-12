# Optimizing Language Models for Argumentative Reasoning

This repository contains and documents key research outputs from a University of Melbourne research project evaluating how large language models can best be optimized to perform argumentative reasoning tasks.

A more detailed paper describing the work is forthcoming, the abstract of which reads:

> Large transformer-based causal language models are capable of strong performance on many natural language processing tasks. Here, we systematically evaluate the performance of the 2.7 billion parameter GPT Neo pretrained language model on 6 argumentative reasoning tasks under 5 different optimization strategies, including prompt programming, soft prompts, and parameter tuning. We report both intrinsic evaluation metrics (perplexity), and extrinsic measures of the coherence of model outputs, as judged by a human rater. With a few exceptions, the rate at which models produced coherent responses ranged from 15-50%. In contrast, human performance (users of the *Kialo* argument mapping platform) ranged from 65-82% coherent, depending on the task. These results suggest that larger, suitably optimized language models may be capable of supporting authors and auditors of natural language argument maps in human-in-the-loop settings. We share our finetuned models and code.

The intended use case is to integrate language models into argument mapping tools. We built a prototype "[argument processor](https://luke-thorburn.github.io/argument-processor/)" that demonstrates how this might look.

## Contents of this repository

This repository contains:

- Code for performing optimization, evaluation, and text generation. (See the `Code/README.md`.)
- The specification of prompt templates used. (The prompt templates are defined in `Code/data.ipynb`, and the prompt templates used by each of the published models are specified in `Code/config/eCONFIG.json`.)
- A set of generated examples that were manually rated, along with coherence ratings from an expert human rater. 100 were generated for each model, but not all are included in this file because some generations were empty or implausibly short after undergoing post-processing, so were not presented to the human rater. (See `GeneratedExamples/final_ratings_30_Jan_2022.json`. This is an array of objects, each of which corresponds to a generated example. The text the model was prompted with is specified in the `in` field, the generated output in the `gen` field, the argumentative reasoning task in the `op` field, and the model used by a unique identifier in the `model` field, which can be cross-referenced with the identifiers specified in `Code/config/gCONFIG.json`.)

Data sharing conditions prevent us from making the data on which the models were trained publicly available, but if you are interested in using it please contact us and we will try to help.

## Links to finetuned models

| Task | Strategy:<br />Soft Prompt | Strategy:<br />Bias-only Finetune | Strategy:<br />Full Finetune |
| ---- | :---------------------: | :---------------------------------: | :---------------------------------: |
| `suggest-reasons` | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-reasons-soft) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-reasons-bias-only) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-reasons-full-finetune) |
| `suggest-objections` | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-objections-soft) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-objections-bias-only) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-objections-full-finetune) |
| `suggest-conclusion` | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-conclusion-soft) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-conclusion-bias-only) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-conclusion-full-finetune) |
| `suggest-intermediary-claims` | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-intermediary-claims-soft) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-intermediary-claims-bias-only) | [Hugging Face](https://huggingface.co/luke-thorburn/suggest-intermediary-claims-full-finetune) |

## Contact

If you have any enquiries about this work, please contact [Luke Thorburn](https://lukethorburn.com/).

## How to cite

For now, please cite this work as:

```
@misc{thorburn2022,
  author = {Thorburn, Luke and Kruger, Ariel},
  title = {\texttt{language-model-optimization}},
  year = {2022},
  publisher = {GitHub},
  institution = {University of Melbourne},
  type = {software},
  url = {https://github.com/Hunt-Laboratory/language-model-optimization}
}
```

## Acknowledgements

This research was funded by the Australian Department of Defence and the Office of National Intelligence under the AI for Decision Making Program, delivered in partnership with the Defence Science Institute in Victoria, Australia.
