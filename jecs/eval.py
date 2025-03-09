import torch   
import matplotlib.pyplot as plt
from model import ShallowMLP
from dataset import JetEnergyCorrectionDataset
def load_model(save_path='/models/best_model_jecs.pt', model_class=None):
    checkpoint = torch.load(save_path,map_location=torch.device('cpu') )

    model = model_class()

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def main():
    model= load_model('best_model_jecs.pt', model_class=lambda: ShallowMLP())
    model.to('cpu')
    model.eval()

    criterion = torch.nn.MSELoss()

    jet_data = JetEnergyCorrectionDataset(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None)
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

    residuals = y_test_tensor.numpy().flatten()-y_pred_tensor.numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()

if __name__ == '__main__':
    main()
