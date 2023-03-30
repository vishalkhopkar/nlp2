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

"""
from QA_Dataset import QADataset
from utils import post_processing
from evaluation_squad import Evaluation
"""
def get_predicted_answers(qa_model, tokenizer, question, context):
    # tokenize the context and question
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    # get the model outputs
    outputs = qa_model(**inputs)

    # get the answer span
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    print(answer)
    return answer

def qa_inference(model,input):
    model.eval()
    start_logits = []
    end_logits = []
    
    with torch.no_grad():  
            #TODO 6: pass the model arguments to the model and store the output
            outputs = model(**input)
            #print(outputs)
            startlog = outputs.start_logits
            endlog = outputs.end_logits

            # TODO 7: Extract the start and end logits by extending `start_logits` and `end_logits`
            start_logits.extend(startlog.cpu().numpy())
            end_logits.extend(endlog.cpu().numpy())

    # TODO 8: Convert the start and end logits to a numpy array (by passing them to `np.array`)
    end_logits = np.array(end_logits)
    start_logits = np.array(start_logits)

    # TODO 9: return start and end logits
    return start_logits, end_logits

def post_processing(raw_dataset, tokenized_dataset, start_logits, end_logits, config):
    
    # Map each example to its features. This is done because an example can have multiple features
    # as we split the context into chunks if it exceeded the max length
    data2features = collections.defaultdict(list)
    for idx, feature_id in enumerate(tokenized_dataset['ID']):
        data2features[feature_id].append(idx)

    # Decode the answers for each datapoint
    predictions = []
    for data in tqdm(raw_dataset):  # data is raw data with attributes: id, title, context, question, answer
        answers = []
        data_id = data["id"]
        context = data["context"]
        #print("data is", data)


        for feature_index in data2features[data_id]: #feature index eg: f indx - 0
            '''flag = 0
            if(flag==0):
              print("f indx -",feature_index)'''

            # TODO 10: Get the start logit, end logit, and offset mapping for each index.
            start_logit = start_logits[feature_index]

            #print("start logit -",start_logits) #eg: start logit - [ 1.0748398  -8.9108715  -9.128406   -9.303913   -9.291563   -8.841147
            end_logit = end_logits[feature_index]
            #print("end logit -",end_logits)
            offset = tokenized_dataset['offset_mapping'][feature_index]


            # TODO 11: Sort the start and end logits and get the top n_best_size logits. - get eg: top 5 start index and top 5 end index. which combination of start and end index has the highest prob?
            # Hint: look at other QA pipelines/tutorials.
            n_best_size = config['n_best_size']
            start_indexes = np.argsort(-1*start_logit)[:n_best_size]
            end_indexes = np.argsort(-1*end_logit)[:n_best_size]

            #print("start indexes len", np.shape(start_indexes) )
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                  isExclude = False
                  # TODO 12: Exclde answers that are not in the context
                  #if(offset not in context) -> continue. else -> store it somewhere?
                  if(offset == (-1,-1)):
                    isExclude = True
                    
                  # TODO 13: Exclude answers if (answer length < 0) or (answer length > max_answer_length)
                  max_answer_length = config['max_answer_length']
                  answer_length = (end_index - start_index)
                  if (answer_length < 0) or (answer_length > max_answer_length):
                    isExclude = True

                  # TODO 14: collect answers in a list. #logit score - prob for each and every word for it being a start index . assigns a number to the value, gives the prob of whether that word is the start of the answer
                  if(isExclude == False):
    
                    #print("context is ", context[start_index:end_index])

                    #calculate the offset va;ues that gives the start and end va;ues of teh tokens 
                    start_offset = offset[start_index][0] #first value of the tuple
                    end_offset = offset[end_index][1] #second value of the tuple

                    answers.append({
                        "text": context[start_offset:end_offset], #eg: ans offset (557, 561, 567, 569)
                        "logit_score": start_logit[start_index] + end_logit[end_index] #eg: logit_score -3.823214
                    })
                    #print("works fine in start idx - {0} and end indx - {1}, feature indx - {2}".format(start_index, end_index,feature_index)) 
                    #print("answers is ",answers)
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predictions.append(
            {
                  "id": data_id, 
                  "prediction_text": best_answer["text"],
                  "no_answer_probability": 0.0 if len(best_answer["text"]) > 0 else 1.0
            }
            ) 
        #print("predictions ", predictions)   
    return predictions


# tokenize the context and question
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    # get the model outputs
    outputs = model(**inputs)

    # get the answer span
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    print(answer)