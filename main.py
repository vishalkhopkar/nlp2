import sys

import torch

from classify import classify
from bool_qa_wrapper import BooleanModel
from wh_qa_wrapper import WhModel
def get_device_type():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using GPU: ", DEVICE)
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU: ", DEVICE)
    return DEVICE


# Press the green button in the gutter to run the script.
# Inputs: <article_file_name> <questions_file_name>
def read_files(article_file_name, questions_file_name):
    with open(article_file_name, 'rb') as f:
        data = f.read()
        article_str = data.decode('utf-8')
    with open(questions_file_name, 'r', encoding='utf-8') as f1:
        questions = f1.readlines()
    return article_str, questions


if __name__ == '__main__':
    n = len(sys.argv)
    if n != 3:
        print("Run this program with inputs <article_file_name> <questions_file_name>")
        exit()
    article_file_name = sys.argv[1]
    questions_file_name = sys.argv[2]
    context, questions = read_files(article_file_name, questions_file_name)
    #print("main context ", context)
    print(questions)
    """
    check type of question
    It is a list, it can be either boolean or WH
    """
    DEVICE = get_device_type()
    boolean_questions, wh_questions, is_bool_list = classify(questions)
    print("is bool list ", is_bool_list)
    boolean_model = BooleanModel(questions=boolean_questions, context=context, device = DEVICE)
    wh_model = WhModel(questions=wh_questions, context=context, device = DEVICE)
    boolean_answers = boolean_model.answers()
    wh_answers = wh_model.answers()

    """
    merge them now
    """
    c1 = 0
    c2 = 0
    N = len(is_bool_list)
    for i in range(N):
        if is_bool_list[i]:
            print(boolean_answers[c1])
            c1 += 1
        else:
            print(wh_answers[c2])
            c2 += 1


