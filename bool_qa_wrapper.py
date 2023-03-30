from bool_qa.bool_qa import *

MODEL_NAME = "shahrukhx01/roberta-base-boolq"

MAX_SEQ_LENGTH = 512

CONFIG = {
    'model_checkpoint': MODEL_NAME,
    "max_length": MAX_SEQ_LENGTH,
    "truncation": 'longest_first',
    "padding": 'max_length',
    "return_overflowing_tokens": True,
    "return_offsets_mapping": True,
    "stride": 100,
    "n_best_size": 20,
    "max_answer_length": 170,
    "batch_size": 128
}


class BooleanModel:
    def __init__(self, questions, context, device):
        self.questions = questions
        self.n = len(questions)
        self.contexts = [context] * self.n
        print("n ",self.n)
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_checkpoint'], max_length=600)
        qa_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, return_dict=True)
        qa_model.to(device)
        self.data = MyOwnBoolQDataset(questions=self.questions, contexts=self.contexts, tokenizer=tokenizer,
                                      config=CONFIG)
        eval_dataloader = DataLoader(self.data, batch_size=CONFIG["batch_size"])

        self.logits = inference(qa_model, eval_dataloader, device)

    def answers(self):
        return post_processing3(self.data, self.logits)
