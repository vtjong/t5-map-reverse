import tqdm
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration

from datasets import load_dataset
from dataset import ReverseEnglishDataset

MAX_LEN_DATASET = 200
LEN_PREFIX = len("Translate from English to reverse-English: ")

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def dataloaders(tokenizer, batch_size):
    train_data = load_dataset("ag_news")["train"]
    test_data = load_dataset("ag_news")["test"]

    train_dataset = ReverseEnglishDataset(train_data, tokenizer)
    test_dataset = ReverseEnglishDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader

def print_metrics(freq, epoch, batch_idx, loss):
    if batch_idx % freq == 0:
        print('epoch: {}, batch_idx: {}, loss: {}'.format(epoch, batch_idx, loss.item()))

def train(tokenizer, model, train_loader, num_epochs, optimizer, freq):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            y = batch['target_ids']
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch['input_ids']
            mask = batch['attention_mask']

            loss = model(input_ids=ids, attention_mask=mask, 
                        decoder_input_ids=y_ids, labels=lm_labels).loss

            print_metrics(freq, epoch, batch_idx, loss)            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def eval(tokenizer, model, test_loader):
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            output_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=MAX_LEN_DATASET,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )

            outputs.extend(
                tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            )

            targets.extend(
                tokenizer.batch_decode(
                    batch['target_ids'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            )
    return outputs, targets


def main():
    parser = argparse.ArgumentParser(description='Finetuning T5 Reverse English')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--print_freq', type=int, default=500, help='print loss every X batch')
    args = parser.parse_args()

    set_seeds(args.seed)
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=MAX_LEN_DATASET + LEN_PREFIX)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train_loader, test_loader = dataloaders(tokenizer, args.batch_size)
    
    train(tokenizer, model, train_loader, args.epochs, optimizer, args.print_freq)
    outputs, labels = eval(tokenizer, model, test_loader)
    pd.DataFrame({'Outputs': outputs, 'Labels': labels}).to_csv('results/test.csv')
    

if __name__ == '__main__':
    main()
