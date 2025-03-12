import torch   
import os
import matplotlib.pyplot as plt
from dataset import JetEnergyCorrectionDataset, load_jet_energy_flow_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

dirname= os.path.dirname(os.path.dirname(__file__))
filename = os.path.join(dirname,'models', 'best_model_jecs.pt')

def load_model(save_path=filename, model_class=None):
    checkpoint = torch.load(save_path,map_location=torch.device('cpu') )

    model = model_class()

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def plot_jecs_by_bins(jets_dataset:JetEnergyCorrectionDataset, model:torch.nn.Module, device:str):
    x_test_0 = jets_dataset.x_test_0
    y_test_0 = jets_dataset.y_test_0

    x_vali_0 = jets_dataset.x_vali_0
    y_vali_0 = jets_dataset.y_vali_0
    scaler_x = jets_dataset.scaler_x
    scaler_y = jets_dataset.scaler_y

    bins=np.array([300,470,600,800,1000,1400,1800,3000])

    x_test_vali = np.concatenate((x_test_0, x_vali_0), axis=0)
    y_test_vali = np.concatenate((y_test_0, y_vali_0), axis=0)

    mask_variable=scaler_x.inverse_transform(x_test_vali)[:,0]

    for i in range(len(bins) - 1):
        mask = (np.array(mask_variable) >= bins[i]) & (np.array(mask_variable) < bins[i + 1])
        
        x_bin = x_test_vali[mask][:, :-1]
        x_bin_tensor = torch.tensor(x_bin, dtype=torch.float32).to(device)

        with torch.no_grad():
            jec_pred_tensor = model(x_bin_tensor).cpu().numpy().flatten()
        
        jec_pred = scaler_y.inverse_transform(jec_pred_tensor.reshape(-1, 1)).flatten()

        x_prime = scaler_x.inverse_transform(x_test_vali[mask])
        y_prime = scaler_y.inverse_transform(y_test_vali[mask].reshape(-1, 1)).flatten()
        pts_bin = np.array(x_prime[:,0])
        gen_pts_bin = np.array(x_prime[:,4])
        jec_bin = np.array(y_prime)
        etas_bin = np.array(x_prime[:,1])

        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].hist(pts_bin / gen_pts_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="red", label="raw")
        ax[0].hist(jec_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="blue", label="JEC factor")
        ax[0].hist(pts_bin * jec_bin / gen_pts_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="purple", label="corrected")
        ax[0].hist(jec_pred, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="green", label="JEC predicted")
        ax[0].hist(pts_bin * jec_pred / gen_pts_bin, bins=np.linspace(0.7, 1.3, 200), histtype="step", color="orange", label="corrected predicted")
        ax[0].set_xlabel("Jet $p_T$ / Gen Jet $p_T$")
        ax[0].set_ylabel("Number of Jets")
        ax[0].set_title(f"Bin {bins[i]}-{bins[i + 1]}")
        ax[0].legend()

        h2 = ax[1].hist2d(pts_bin * jec_bin / gen_pts_bin, etas_bin, 
                        bins=[np.linspace(0.7, 1.3, 200), np.linspace(-4, 4, 200)], cmap='Reds', alpha=1, label="corrected")

        h3 = ax[1].hist2d(pts_bin * jec_pred / gen_pts_bin, etas_bin, 
                        bins=[np.linspace(0.7, 1.3, 200), np.linspace(-4, 4, 200)], cmap='Greens', alpha=0.6, label="corrected predicted")

        ax[1].set_xlabel("Jet $p_T$ / Gen Jet $p_T$")
        ax[1].set_ylabel("Jet $\eta$")
        ax[1].set_title(f"Bin {bins[i]}-{bins[i + 1]} pT std={np.std(pts_bin / gen_pts_bin):.3f} eta std={np.std(etas_bin):.3f}")

        plt.colorbar(h2[3], ax=ax[1], label='corrected')
        plt.colorbar(h3[3], ax=ax[1], label='corrected predicted')

        plt.show()


def run(jet_data=None):
    from j_model import ShallowMLP
    x=ShallowMLP()
    model= load_model(filename, model_class=lambda: ShallowMLP())
    model.to('cpu')
    model.eval()

    criterion = torch.nn.MSELoss()
    if not jet_data:
        jet_data = load_jet_energy_flow_dataset(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None)
    x_test_0 = jet_data.x_test_0
    y_test_0 = jet_data.y_test_0
    x_test_tensor = torch.tensor(x_test_0, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_0, dtype=torch.float32)

    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor[:,:-1])
        test_loss = criterion(y_pred_tensor, y_test_tensor.unsqueeze(1)).item()
    print(f"Test MSE: {test_loss}")

    plt.figure(figsize=(8, 8))
    plt.hist2d(y_test_tensor.numpy().flatten(), y_pred_tensor.numpy().flatten(), bins=150, cmap='viridis')
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

    plot_jecs_by_bins(jet_data, model, 'cpu')


if __name__ == '__main__':
    run()
