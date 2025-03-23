import torch   
import os
import matplotlib.pyplot as plt
from dataset import JetEnergyCorrectionDataset, load_jet_energy_flow_dataset, load_scaler
import numpy as np
from sklearn.preprocessing import StandardScaler

dirname= os.path.dirname(os.path.dirname(__file__))
filename = os.path.join(dirname,'models','current_trial', 'best_model_jecs.pt')

def load_model(save_path=filename, model_class=None):
    checkpoint = torch.load(save_path,map_location=torch.device('cpu') )

    model = model_class()

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def run(x,y, model=None):
    from j_model import ShallowMLP
    from matplotlib.colors import LogNorm
    from sklearn.preprocessing import StandardScaler
    if model is None:
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
        y_pred_tensor = model(x_test_tensor)
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

    plot_jecs_by_bins(x,y, model, 'cpu',scaler_x,scaler_y)

def plot_jecs_by_bins(x,y, model:torch.nn.Module, device:str,scaler_x:StandardScaler,scaler_y:StandardScaler):

    scaler_x,scaler_y = load_scaler()
    mask_variable=x[:,0]

    x = scaler_x.transform(x)   

    bins=np.array([0,50,100,150,250,550,2000])

    for i in range(len(bins) - 1):
        mask = (np.array(mask_variable) >= bins[i]) & (np.array(mask_variable) < bins[i + 1])

        x_bin = x[mask]
        y_bin= y[mask]
        x_bin_tensor = torch.tensor(x_bin, dtype=torch.float32).to(device)

        with torch.no_grad():
            jec_pred_tensor = model(x_bin_tensor).cpu().numpy().flatten()

        jec_pred = scaler_y.inverse_transform(jec_pred_tensor.reshape(-1, 1)).flatten()

        x_prime = scaler_x.inverse_transform(x_bin)

        pts_bin = np.array(x_prime[:, 0])
        jec_bin = np.array(y_bin)
        etas_bin = np.array(x_prime[:, 1])

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].hist(jec_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="blue", label="JEC factor")
        ax[0].hist(jec_pred, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="green", label="JEC predicted")
        ax[0].set_xlabel("jec")
        ax[0].set_ylabel("Number of Jets")
        ax[0].set_title(f"Bin {bins[i]}-{bins[i + 1]}")
        ax[0].legend()

        ax[1].hist(pts_bin, bins=200, histtype="step", color="red", label="raw")
        ax[1].hist(pts_bin * jec_bin, bins=200, histtype="step", color="purple", label="corrected")
        ax[1].hist(pts_bin * jec_pred, bins=200, histtype="step", color="orange", label="corrected predicted")
        ax[1].set_xlabel("Jet $p_T$")
        ax[1].set_ylabel("Number of Jets")
        ax[1].set_title(f"Bin {bins[i]}-{bins[i + 1]}")
        ax[1].legend()

        h2 = ax[2].hist2d(pts_bin * jec_bin, etas_bin, bins=[200, 200], cmap='Reds', alpha=1, label="corrected")
        h3 = ax[2].hist2d(pts_bin * jec_pred, etas_bin, bins=[200, 200], cmap='Greens', alpha=0.6, label="corrected predicted")

        ax[2].set_xlabel("Jet $p_T$")
        ax[2].set_ylabel("Jet $\eta$")
        ax[2].set_title(f"Bin {bins[i]}-{bins[i + 1]} pT std={np.std(pts_bin):.3f} eta std={np.std(etas_bin):.3f}")

        plt.colorbar(h2[3], ax=ax[2], label='corrected')
        plt.colorbar(h3[3], ax=ax[2], label='corrected predicted')

        plt.show()

    '''

    for i in range(len(bins) - 1):
        mask = (np.array(mask_variable) >= bins[i]) & (np.array(mask_variable) < bins[i + 1])
        
        x_bin = x[mask][:, :-1]
        x_bin_tensor = torch.tensor(x_bin, dtype=torch.float32).to(device)

        with torch.no_grad():
            jec_pred_tensor = model(x_bin_tensor).cpu().numpy().flatten()
        
        jec_pred = scaler_y.inverse_transform(jec_pred_tensor.reshape(-1, 1)).flatten()

        x_prime = scaler_x.inverse_transform(x[mask])
        y_prime = scaler_y.inverse_transform(y[mask].reshape(-1, 1)).flatten()
        pts_bin = np.array(x_prime[:,0])
        jec_bin = np.array(y_prime)
        etas_bin = np.array(x_prime[:,1])

        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.hist(pts_bin/ max(pts_bin), bins=np.linspace(0.7, 1.3, 200), histtype="step", color="red", label="raw")
        ax.hist(jec_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="blue", label="JEC factor")
        ax.hist(pts_bin * jec_bin / max(pts_bin), bins=np.linspace(0.7, 1.3, 200), histtype="step", color="purple", label="corrected")
        ax.hist(jec_pred, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="green", label="JEC predicted")
        ax.hist(pts_bin * jec_pred / max (pts_bin), bins=np.linspace(0.7, 1.3, 200), histtype="step", color="orange", label="corrected predicted")
        ax.set_xlabel("Jet $p_T$ ")
        ax.set_ylabel("Number of Jets")
        ax.set_title(f"Bin {bins[i]}-{bins[i + 1]}")
        ax.legend()
        plt.show()
'''

if __name__ == '__main__':
    pass
