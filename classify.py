import nltk


# Download the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

allowed = {'MD', 'VB', 'VBZ', 'VBD', 'VBP'}


# Tokenize the text into words

def classify(questions):
    boolean_questions = []
    wh_questions = []
    is_boolean_list = []
    for question in questions:
        question = question.lower()
        tokens = nltk.word_tokenize(question)
        pos_tags = nltk.pos_tag(tokens)
        print(pos_tags)
        if pos_tags[0][1] in allowed:
            boolean_questions.append(question)
            is_boolean_list.append(True)
        else:
            wh_questions.append(question)
            is_boolean_list.append(False)
    return boolean_questions, wh_questions, is_boolean_list
