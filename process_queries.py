import torch
from transformers import BertTokenizer, BertForQuestionAnswering

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')


def get_model():
    model = torch.load('bert_finetuned_model', map_location=device)
    model.eval()
    
    return model

def get_answer(context,query, model, tokenizer):
    inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(outputs[1]) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

def load_model(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def main():
    model = load_model('bert_finetuned_model.pth')
    
    context = """ Queen are a British rock band formed in London in 1970. Their classic line-up was Freddie Mercury (lead vocals, piano), 
            Brian May (guitar, vocals), Roger Taylor (drums, vocals) and John Deacon (bass). Their earliest works were influenced 
            by progressive rock, hard rock and heavy metal, but the band gradually ventured into more conventional and radio-friendly 
            works by incorporating further styles, such as arena rock and pop rock. """

    queries = [ 'When did Queen found?',
                'Who were the basic members of Queen band?',
                'What kind of band they are?']
    
    for querie in queries:
        answer = get_answer(querie, context, model, tokenizer)
        print(answer)

if __name__ == '__main__':
    main()