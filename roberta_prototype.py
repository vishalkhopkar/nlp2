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
from evaluation_squad import Evaluation, qa_inference, Extract_Write_Answers, Evaluate_On_Squad
from qa_prototype_utils import get_predicted_answers #import qa_inference, post_processing

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))
    #Cudnn optimized version of cuda
    torch.backends.cudnn.benchmark = False #if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.enabled = False #controls whether cuDNN is enabled
    torch.backends.cudnn.deterministic = True #if True, causes cuDNN to only use deterministic convolution algorithms

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) #accessing env variable called pythonhashseed
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #set seed value on all GPUs

def Save_Pretrained_Models():
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    model.save_pretrained('./saved_models/roberta-base-model')
    tokenizer.save_pretrained('./saved_models/roberta-base-tokenizer')


#if __name__ == "__main__":
def main_func(questions, context):
    #inputs - 1 context, multiple questions, pass each question
    #Placeholder example context and questions given for now
    """
    context = "The quick brown fox jumps over the lazy dog."
    questions = ["What does the fox jump over?", "What color is the fox?"]
    """

    #TODO: Fetch the actual context and questions

    #Set device - CPU vs Cuda
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using GPU: ", DEVICE)
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU: ", DEVICE)

    #set random seed
    SEED = 0
    set_random_seed(SEED)

    #Run only once or comment this line if models are already saved in the server
    Save_Pretrained_Models()

    #Set config
    config = {
    'model_checkpoint': "roberta-base",
    "max_length": 400, #ideal range 300 - 512. input seq will have question + context/paragraph
    "truncation": "only_second", #only second
    "padding": True,
    "return_overflowing_tokens": True, #Whether to include overflowing token information in the returned dictionary when true
    "return_offsets_mapping": True, #index of start and end char in original input for each token
    "stride": 128,
    "n_best_size": 10,
    "max_answer_length": 500,
    "batch_size": 96
    }

    # load the saved model and tokenizer from the directory
    model = RobertaForQuestionAnswering.from_pretrained('saved_models/roberta-base-model')
    tokenizer = AutoTokenizer.from_pretrained('saved_models/roberta-base-tokenizer')

    qa_model = model.to(DEVICE)

    #UNCOMMENT BELOW LINES ONLY FOR EVALUATING ON SQUAD:
    
    #1 download datasets
    #datasets = load_dataset("squad_v2")

    #2 Evaluate on Squad
    '''Evaluate_On_Squad(qa_model=qa_model,
                      datasets=datasets,
                      tokenizer=tokenizer,
                      config=config,
                      DEVICE=DEVICE)'''

    #UNCOMMENT BELOW ONLY IF ANY FINE TUNING DONE TO THE MODEL
    #save fine tuned model 
    '''output_dir = './fine_tuned_model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)'''

    #RUNNING THE PROTOTYPE
    final_answers = []
    for question in questions:
        answer = get_predicted_answers(qa_model= qa_model,
                                       tokenizer= tokenizer,
                                       question= question,
                                       context= context)
        
        final_answers.append(answer)
    print(final_answers)
        


    





