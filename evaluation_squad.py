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

from QA_Dataset import QADataset
from utils import post_processing
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate
)

def Evaluate_On_Squad(qa_model, datasets, tokenizer, config, DEVICE):

    #start training - call the roberta model file
    eval_dataset = QADataset(
    data=datasets['validation'], #are we taking only the validation dataset?
    tokenizer=tokenizer, #this step is just to tokenize the data based on the configuration?
    config=config
    )
    eval_data = eval_dataset.data
    eval_features = eval_dataset.tokenized_data

    #Initialize data loader
    eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config["batch_size"]
    )
    len(eval_dataloader)

    #Evaluation
    start_logits, end_logits = qa_inference(qa_model, eval_dataloader, DEVICE)

    #Post-processing
    predicted_answers = post_processing(
    raw_dataset=eval_data, 
    tokenized_dataset=eval_features,
    start_logits=start_logits,
    end_logits=end_logits,
    config = config
    )

    gold_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_data][:len(eval_data)]

    assert len(predicted_answers) == len(gold_answers)

    #Start Evaluation - on test dataset
    test_dataloader, test_raw, test_tokenized = Evaluation(predicted_answers, gold_answers, config, tokenizer)
    
    #Run Inference on evaluation
    Extract_Write_Answers(qa_model, test_dataloader, test_raw, test_tokenized)

def qa_inference(model, data_loader, DEVICE):
    model.eval()
    start_logits = []
    end_logits = []
    for step, batch in enumerate(tqdm(data_loader, desc="Inference Iteration")):
        with torch.no_grad():
            model_kwargs = {
                'input_ids': batch['input_ids'].to(DEVICE, dtype=torch.long),
                'attention_mask': batch['attention_mask'].to(DEVICE, dtype=torch.long)
            }    

            # TODO 6: pass the model arguments to the model and store the output
            outputs = model(**model_kwargs)
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

def Extract_Write_Answers(qa_model, test_dataloader, test_raw, test_tokenized):
    #Call evaluation method

    #Run the evaluation
    start_logits_test, end_logits_test = qa_inference(qa_model, test_dataloader)

    predicted_answers_test = post_processing(
    raw_dataset=test_raw, 
    tokenized_dataset=test_tokenized,
    start_logits=start_logits_test,
    end_logits=end_logits_test
    )

    with open('blind_test_predictions.json', 'w') as f:
        json.dump(predicted_answers_test, f)


def Evaluation(predicted_answers,gold_answers,config,tokenizer):
    eval_metric = load("squad_v2")
    eval_results = eval_metric.compute(predictions=predicted_answers, references=gold_answers)

    print("----------------SQUAD EVALUATION RESULTS-------------------")
    print(eval_results)

    with open('squad_results.json', 'w') as f:
        json.dump(eval_results, f)

    test_dataset = load_dataset("csv", data_files="blind_test_set.csv", split='train')

    index = random.randint(0, len(test_dataset))
    datapoint = test_dataset[index]
    print(f"index: {index}\n")
    for column, info in datapoint.items():
        print(f"\n{column}:\t{info}")

    Test_dataset = QADataset(
    data= test_dataset, #are we taking only the validation dataset?
    tokenizer=tokenizer,
    config=config
    )

    # TODO 18: Define the dataloader for the test set 
    test_dataloader = DataLoader(
        Test_dataset,
        batch_size=config["batch_size"]
    )
    len(test_dataloader)

    test_raw = Test_dataset.data
    test_tokenized = Test_dataset.tokenized_data

    return test_dataloader, test_raw, test_tokenized

