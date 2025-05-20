import streamlit as st
import torch
from transformers import BertTokenizer
from process_queries import load_model, get_answer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
def main():
    st.title('Question Answering with Fine-Tuned BERT')
    st.write('Enter a context and a question, and the model will return an answer.')

    model = load_model('bert_finetuned_model.pth')

    context = st.text_area('Enter context here:', height=200)
    question = st.text_input('Enter question:')

    if st.button('Get Answer'):
        if context.strip() and question.strip():
            answer = get_answer(question, context, model, tokenizer)
            st.subheader('Answer:')
            st.write(answer)
        else:
            st.warning('Please enter both a context and a question.')
        
if __name__ == '__main__':
    main()
