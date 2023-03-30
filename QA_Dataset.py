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
    
    def __init__(
        self,
        data,
        tokenizer,
        config
    ):

        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.tokenized_data = self.tokenizer(
            self.data["question"],
            self.data["context"],
            max_length=self.config["max_length"],
            stride=self.config["stride"],
            truncation=self.config["truncation"],
            padding=self.config["padding"],
            return_overflowing_tokens=self.config["return_overflowing_tokens"],
            return_offsets_mapping=self.config["return_offsets_mapping"],
            return_attention_mask=True,
            add_special_tokens=True
        )
        #print(self.tokenized_data)
        
        example_ids = []
        for i, sample_mapping in enumerate(tqdm(self.tokenized_data["overflow_to_sample_mapping"])): #Q: are we iterating over the dataset or each token?
            #print(self.tokenized_data["offset_mapping"][0])
            example_ids.append(self.data["id"][sample_mapping])

            sequence_ids = self.tokenized_data.sequence_ids(i) 
            offset_mapping = self.tokenized_data["offset_mapping"][i]
            #print("offset mapping")
            #print(offset_mapping)
            
            # TODO 3: set the offset mapping of the tokenized data at index i to (-1, -1) 
            # if the token is not in the context
            self.tokenized_data["offset_mapping"][i] = [offset if sequence_ids[k] == 1 else tuple([-1,-1]) for k, offset in enumerate(offset_mapping)]
            
            '''j = 0
            for tok in self.tokenized_data.sequence_ids(i):
              #print("sequence id is ", tok)
              if(tok != 1):
                self.tokenized_data["offset_mapping"][j] = (-1, -1)  #Q: WILL EVERY TOKEN IN THE SEQUENCE HAVE AN OFFSET MAPPING?
                #print("offset mapping is ", self.tokenized_data["offset_mapping"][l])
                j = j + 1    '''  

        self.tokenized_data["ID"] = example_ids
        #print(example_ids)
        
        
        
    def __len__(self):
      # TODO 4: define the length of the dataset equal to total number of unique features (not the total number of datapoints) Q: WHAT IS MEANT BY UNIQUE FEATURES?
      return len(self.tokenized_data['input_ids'])
    
    
    
    def __getitem__(self, index: int):
      # TODO 5: Return the tokenized dataset at the given index. Convert the various inputs to tensor using torch.tensor Q: VARIOUS INPUTS?
      return {
            'input_ids': torch.tensor(self.tokenized_data['input_ids'][index], dtype = torch.long) ,
            'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][index], dtype = torch.long),
            'offset_mapping': torch.tensor(self.tokenized_data['offset_mapping'][index], dtype = torch.long),
            'example_id': self.tokenized_data['ID'][index]
        }
