import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import collections
from tqdm import tqdm
import os
import json

import torch
from torch.utils.data import (
DataLoader,
Dataset
)

from datasets import load_dataset
from evaluate import load

from transformers import (
AutoTokenizer,
AutoModelForQuestionAnswering,
RobertaForQuestionAnswering,
Trainer, 
TrainingArguments
)

class QADataset(Dataset):
    
    # constructor
    def __init__(
        self,
        questions,
        contexts,
        tokenizer,
        config
    ):

        self.config = config
        self.questions = questions
        self.contexts = contexts
        self.ids = [i for i in range(len(self.questions))]
        self.tokenizer = tokenizer
        self.tokenized_data = self.tokenizer(
            self.questions,
            self.contexts,
            max_length=self.config["max_length"],
            stride=self.config["stride"],
            truncation=self.config["truncation"],
            padding=self.config["padding"],
            return_overflowing_tokens=self.config["return_overflowing_tokens"],
            return_offsets_mapping=self.config["return_offsets_mapping"],
            return_attention_mask=True,
            add_special_tokens=True
        )
        
        
        print("len of samp mapping ", len(self.tokenized_data["overflow_to_sample_mapping"]))
        print("self.ids length ", len(self.ids))
        '''for i in range(10):
            print(self.ids[i])'''
        for i, sample_mapping in enumerate(tqdm(self.tokenized_data["overflow_to_sample_mapping"])):
                       
            sequence_ids = self.tokenized_data.sequence_ids(i)
            offset_mapping = self.tokenized_data["offset_mapping"][i]
           
            
            for j, id in enumerate(sequence_ids):
                if id == 0:
                    self.tokenized_data["offset_mapping"][i][j] = (-1, -1)
            
            
        self.tokenized_data["ID"] = self.tokenized_data["overflow_to_sample_mapping"]
        print("tokenized_dataset['ID'] length ", len(self.tokenized_data['ID']))
        
    def __len__(
        self
    ):
        
        return len(self.tokenized_data['input_ids'])
    
    
    def __getitem__(
        self,
        index: int
    ):
        
        return {
            'input_ids': torch.tensor(self.tokenized_data['input_ids'][index]),
            'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][index]),
            'offset_mapping': torch.tensor(self.tokenized_data['offset_mapping'][index]),
            'example_id': self.tokenized_data["ID"][index],
        }

