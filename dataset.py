from torch.utils.data import Dataset
import torch
import sys

def process_string(string):
    return ' '.join(string.split())

class ReverseEnglishDataset(Dataset):

    def __init__(self, dataset, tokenizer):
        self.data = dataset["text"]
        self.task_prefix = "Translate from English to reverse-English: "
        self.max_len_dataset = 200
        self.len_prefix = len(self.task_prefix) 
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = self.data[index]
        reversed_text = input_text[::-1]
        input_text = process_string(input_text)
        reversed_text = process_string(reversed_text)

        input = self.tokenizer.batch_encode_plus([self.task_prefix + input_text], 
                                                    max_length=self.max_len_dataset + self.len_prefix, 
                                                    pad_to_max_length=True,
                                                    return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([reversed_text], 
                                                    max_length=self.max_len_dataset, 
                                                    pad_to_max_length=True,
                                                    return_tensors='pt')

        input_ids = input['input_ids'].squeeze()
        attention_mask = input['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'input_text': input_text,
            'target_text': reversed_text,
            'input_ids': input_ids.to(dtype=torch.long),
            'attention_mask': attention_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }