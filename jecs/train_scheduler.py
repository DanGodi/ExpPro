import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import JetEnergyCorrectionDataset
import os

dirname= os.path.dirname(os.path.dirname(__file__))
filename = os.path.join(dirname,'models','current_trial', 'best_model_jecs.pt')

class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss

            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")



            torch.save({'model' : model,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss function': criterion,
                'loss': current_valid_loss,
                }, filename)

            return 1
        else:
          return 0



def train(model:nn.Module, jet_dataset: JetEnergyCorrectionDataset,criterion=nn.MSELoss(), num_epochs:int=40,device:str='cpu'):
    model=model.to(device)
    train_dl = jet_dataset.train_dl
    vali_dl = jet_dataset.vali_dl
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1,min_lr= 0.001 , verbose=True)
    
    save_best_model = SaveBestModel()

    train_losses = []
    vali_losses = []

    for epoch in range(num_epochs):
        #training
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        
        train_loss = running_loss / len(train_dl.dataset)
        train_losses.append(train_loss)

        scheduler.step(train_loss)
        
        #validation step
        model.eval()
        running_vali_loss = 0.0
        with torch.no_grad():
            for xb, yb in vali_dl:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb.unsqueeze(1))
                running_vali_loss += loss.item() * xb.size(0)
        
        vali_loss = running_vali_loss / len(vali_dl.dataset)
        vali_losses.append(vali_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.9f}, Validation Loss: {vali_loss:.9f}")

        # Save the model
        save_best_model(vali_loss, epoch, model, optimizer, criterion)


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), vali_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    from jecs.j_model import ShallowMLP

    model = ShallowMLP()

    jet_dataset = JetEnergyCorrectionDataset()

    train(model, jet_dataset)