from roberta_prototype import *

class WhModel:
    def __init__(self, questions, context, device):
        self.answers = main_func(questions, context)

    def answers(self):

        return self.answers

