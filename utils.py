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

def post_processing(raw_dataset, tokenized_dataset, start_logits, end_logits, config):
    # Map each example to its features. This is done because an example can have multiple features
    # as we split the context into chunks if it exceeded the max length
    data2features = collections.defaultdict(list)
    n_best_size = config['n_best_size']
    for idx, feature_id in enumerate(tokenized_dataset['ID']):
        data2features[feature_id].append(idx)
    
    # Decode the answers for each datapoint
    predictions = []
    for i in range(len(raw_dataset.questions)):
        #print(i)
        answers = []
        data_id = i
        #print(data_id)
        context = raw_dataset.contexts[i]

        for feature_index in data2features[data_id]:
            #print("i = ", i, " feature index = ", feature_index)
            # TODO 10: Get the start logit, end logit, and offset mapping for each index.
            start_logit_arr = start_logits[feature_index]
            end_logit_arr = end_logits[feature_index]
            
            offset_mapping = tokenized_dataset['offset_mapping'][feature_index]
            # TODO 11: Sort the start and end logits and get the top n_best_size logits.
            start_indexes = np.argsort(start_logit_arr)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit_arr)[-1 : -n_best_size - 1 : -1].tolist()
            offset_mapping = tokenized_dataset['offset_mapping'][feature_index]
            # Hint: look at other QA pipelines/tutorials.
            
            for start_index in start_indexes:
                if offset_mapping[start_index][0] != -1 and offset_mapping[start_index][1] != -1:
                    for end_index in end_indexes:
                        #if end_index > start_index:
                        if offset_mapping[end_index][0] != -1 and offset_mapping[end_index][1] != -1:                            
                            #if offset_mapping[end_index][1] >= offset_mapping[end_index][0]:                            
                            #if offset_mapping[end_index][1] - offset_mapping[start_index][0] <= config['max_answer_length']:
                            answer_length = end_index - start_index + 1
                            if answer_length > 0 and answer_length <= config['max_answer_length']:
                                logit_score = start_logit_arr[start_index] + end_logit_arr[end_index]                                
                                answers.append(
                                    {
                                        "text": context[offset_mapping[start_index][0]:offset_mapping[end_index][1]],
                                        #"text": tokenizer.decode(tokenized_dataset["input_ids"][i][start_index - 1 : end_index + 1]),
                                        "logit_score": logit_score,
                                    }
                                )
                                #print(answers)        
                
        best_answer = max(answers, key=lambda x: x["logit_score"])
        
        predictions.append(
            {
                "id": data_id, 
                "prediction_text": best_answer["text"],
                "no_answer_probability": 0.0 if len(best_answer["text"]) > 0 else 1.
                #"logit_score": best_answer["logit_score"]
            }
        )
    return predictions