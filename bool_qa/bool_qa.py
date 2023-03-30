import json
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import (
    DataLoader,
    Dataset
)

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification

from tqdm import tqdm


class MyOwnBoolQDataset(Dataset):
    # constructor
    def __init__(self, questions, contexts, tokenizer, config):
        self.config = config
        self.questions = questions
        self.tokenizer = tokenizer
        print(contexts)
        self.contexts = contexts
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
        self.inverse_mapping = np.ndarray(shape=(len(self.questions)), dtype=list)
        for i, mapping in enumerate(self.tokenized_data["overflow_to_sample_mapping"]):
            if self.inverse_mapping[mapping] is None:
                self.inverse_mapping[mapping] = []
            self.inverse_mapping[mapping].append(i)

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

    def __getitem__(self, index: int):
        return {
            'input_ids': torch.tensor(self.tokenized_data['input_ids'][index]),
            'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][index]),
            'offset_mapping': torch.tensor(self.tokenized_data['offset_mapping'][index])
        }


def predict(logits):
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)
    return True if (proba_yes >= proba_no) else False


def inference(model, data_loader, device):
    model.eval()
    first_done = False
    for step, batch in enumerate(tqdm(data_loader, desc="Inference Iteration")):

        with torch.no_grad():
            model_kwargs = {
                'input_ids': batch['input_ids'].to(device, dtype=torch.long),
                'attention_mask': batch['attention_mask'].to(device, dtype=torch.long)
            }
            output = model(**model_kwargs)[0].cpu().detach().numpy()
            # print(output)

            if not first_done:
                logits = output
                first_done = True
            else:
                logits = np.concatenate((logits, output), axis=0)

    return logits


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtracting the max value of x to avoid numerical instability
    return e_x / e_x.sum(axis=0)


def post_processing3(boolq_dataset, full_logits):
    predictions = []
    inverse_mapping = boolq_dataset.inverse_mapping

    for i in range(len(inverse_mapping)):
        logit_ids = inverse_mapping[i]
        first_idx = logit_ids[0]
        last_idx = logit_ids[-1]
        logits_sub_array = full_logits[first_idx: last_idx + 1]

        probabilities = np.array([softmax(logits) for logits in logits_sub_array])
        max_probs = np.argmax(probabilities)
        max_probs_col = max_probs % probabilities.shape[1]
        best_answer = True if max_probs_col == 1 else False
        predictions.append(best_answer)
    return predictions
