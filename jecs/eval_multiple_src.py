import torch   
import os
import matplotlib.pyplot as plt
from dataset import JetEnergyCorrectionDataset, load_jet_energy_flow_dataset, load_scaler
import numpy as np
from sklearn.preprocessing import StandardScaler

dirname= os.path.dirname(os.path.dirname(__file__))
filename = os.path.join(dirname,'models', 'best_model_jecs.pt')

def load_model(save_path=filename, model_class=None):
    checkpoint = torch.load(save_path,map_location=torch.device('cpu') )

    model = model_class()

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def run(x,y):
    from j_model import ShallowMLP
    from matplotlib.colors import LogNorm
    from sklearn.preprocessing import StandardScaler
    model= load_model(filename, model_class=lambda: ShallowMLP())
    model.to('cpu')
    model.eval()

    criterion = torch.nn.MSELoss()
    
    scaler_x,scaler_y = load_scaler()

    x_test_0 = scaler_x.transform(x)   
    y_test_0 = scaler_y.transform(y.reshape(-1, 1)).flatten()

    x_test_tensor = torch.tensor(x_test_0, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_0, dtype=torch.float32)

    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor[:,:-1])
        test_loss = criterion(y_pred_tensor, y_test_tensor.unsqueeze(1)).item()
    print(f"Test MSE: {test_loss}")


    plt.figure(figsize=(8, 8))
    plt.hist2d(y_test_tensor.numpy().flatten(), y_pred_tensor.numpy().flatten(), bins=100,cmap='viridis', norm=LogNorm())
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Results')
    plt.plot([min(y_test_tensor.numpy().flatten()), max(y_test_tensor.numpy().flatten())], [min(y_test_tensor.numpy().flatten()), max(y_test_tensor.numpy().flatten())], color='red')  # Perfect fit line
    plt.colorbar(label='Counts')
    plt.show()

    # Residuals
    residuals = y_test_tensor.numpy().flatten()-y_pred_tensor.numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()


if __name__ == '__main__':
    pass
