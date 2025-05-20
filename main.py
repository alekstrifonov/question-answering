from squad_dataset import get_datasets
import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from torch.optim import AdamW

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, epochs, optim, train_loader, val_loader):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        loss_of_epoch = 0
        
        for batch_idx, batch in enumerate(train_loader): 
            optim.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            loss_of_epoch += loss.item()
            
            if (batch_idx  % 100) == 0:
                print(f'Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.2f}')
            
        train_losses.append(loss_of_epoch / len(train_loader))
        
        model.eval()
        val_loss_of_epoch = 0
        
        print('\nEvaluating\n')
        for batch_idx, batch in enumerate(val_loader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss  
                val_loss_of_epoch += loss.item()
                
            if (batch_idx  % 100) == 0:
                print(f'Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.2f}')
        
        val_losses.append(val_loss_of_epoch / len(val_loader))
        
        print(f'Epoch {epoch+1}')


def main(): 
    train_dataset, val_dataset = get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
    
    optim = AdamW(model.parameters(), lr=0.005)
    
    train(model, epochs=3, optim=optim, train_loader=train_dataloader, val_loader=val_dataloader)
    
    torch.save(model.state_dict(), 'bert_finetuned_model.pth')

if __name__ == '__main__':
    main()
