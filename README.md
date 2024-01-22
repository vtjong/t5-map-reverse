# Reverse English via Finetuning T5

This repository contains code to finetune the T5 encoder-decoder model, presented in "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. The model comes in 5 sizes: t5-small, t5-base, t5-large, t5-3b, t5-11b, as presented in the above paper. For this code, we take inspiration from https://huggingface.co/docs/transformers/model_doc/t5.

## Experiment

We use the `ag_news` dataset from HuggingFace, which contains input text of English headlines from news articles and label of category classification for over 1 million news articles, and take only the input text as our input and the reversed text as our output, to finetune a T5 model of size t5-base. We use the following as our prefix——"Translate from English to reverse-English: ". Below are the training hyperparameters utilized:

- **Number of Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: 3e-4
- **Dataset Split**:In-built split from HuggingFace
- **Loss**: CE Loss

To run, navigate to `main.py`.

## Results

Test results can be found in `results/test.csv`. After training for 10 epochs, loss is 0.00753.
