import json
import torch
from transformers import AutoTokenizer

import time

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def load_squad_data(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def start_end(answers, contexts):
    for answer, text in zip(answers, contexts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        
        end_idx = start_idx + len(real_answer) # Get the real end index

       
        if text[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        elif text[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif text[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2
            
def tokenize(contexts, queries, tokenizer):
    return tokenizer(contexts, queries, truncation=True, padding=True)

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
            
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
                
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def get_datasets(): 
    train_contexts, train_queries, train_answers = load_squad_data('Data/train-v2.0.json')
    val_contexts, val_queries, val_answers = load_squad_data('Data/dev-v2.0.json')
    
    print(len(train_contexts))
    print(len(train_queries))
    print(len(train_answers))
    
    print(len(val_contexts))
    print(len(val_queries))
    print(len(val_answers))
    
    start_end(train_answers, train_contexts)
    start_end(val_answers, val_contexts)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenize(train_contexts, train_queries, tokenizer)
    val_encodings = tokenize(val_contexts, val_queries, tokenizer)
    
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)
    
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    
    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = get_datasets()