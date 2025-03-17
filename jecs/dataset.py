import energyflow as ef
import numpy as np
import torch
import uproot
import awkward as ak
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os

dirname= os.path.dirname(os.path.dirname(__file__))


def load_energy_flow(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None):
    pts = []
    etas = []
    area=[]
    npvs=[]
    jec=[]
    data = ef.mod.load(amount=amount, cache_dir=cache_dir,
                        dataset=dataset, subdatasets=subdatasets)
    
    for j in range(len(data.jet_pts)):

        pts.append(data.jet_pts[j])
        etas.append(data.jet_etas[j])
        area.append(data.jet_areas[j])
        npvs.append(data.npvs[j])
        jec.append(data.jecs[j])
    
    return pts, etas, area, npvs, jec

def _dump_root_jets(fp, max_N_jets_per_event=10, lib='np'):
    with uproot.open(fp) as f:
        tree = f['pfJetsAk5/Events']
        branches = ['jet_e', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass', 'jet_btag', 'jet_area', 'jet_rho', 'jet_npv', 'jet_jec', 'corr_jet_pt']
        jet_dict = tree.arrays(branches, library=lib)
        return jet_dict



def load_root_files(filenames):
    for f in filenames:
        jets = _dump_root_jets(f)
        pts = np.hstack(jets['jet_pt'])
        etas = np.hstack(jets['jet_eta'])
        areas = np.hstack(jets['jet_area'])
        npvs = np.hstack(jets['jet_npv'])
        jec = np.hstack(jets['jet_jec'])
        yield pts, etas, areas, npvs, jec
        


class JetEnergyCorrectionDataset(TensorDataset):
    def __init__(self, pts, etas, area, npvs, jec, batch_size=512,device:str='cpu'):
        self.__load(pts, etas, area, npvs,  jec, batch_size,device)

    def __len__(self):
        return len(self.train_dl.dataset)+len(self.vali_dl.dataset)+len(self.x_test_0)
    
    def __getitem__(self, idx):
        if idx < len(self.train_dl.dataset):
            return self.train_dl.dataset[idx]
        elif idx < len(self.train_dl.dataset) + len(self.vali_dl.dataset):
            return self.vali_dl.dataset[idx - len(self.train_dl.dataset)]
        else:
            idx -= len(self.train_dl.dataset) + len(self.vali_dl.dataset)
            return self.x_test_0[idx]
        

    def __load(self,pts, etas, area, npvs, jec, batch_size,device):
        x=np.transpose(np.array([pts,etas,area,npvs]))
        y=np.array(jec)

        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(x, y, test_size=0.2, shuffle=True)
        x_train_0,x_vali_0,y_train_0,y_vali_0 = train_test_split(x_train_0 , y_train_0 , test_size=0.4, shuffle=True)

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        x_train_0 = self.scaler_x.fit_transform(x_train_0)
        self.x_vali_0 = self.scaler_x.transform(x_vali_0)
        self.x_test_0 = self.scaler_x.transform(x_test_0)

        y_train_0 = self.scaler_y.fit_transform(y_train_0.reshape(-1, 1)).flatten()
        self.y_vali_0 = self.scaler_y.transform(y_vali_0.reshape(-1, 1)).flatten()
        self.y_test_0 = self.scaler_y.transform(y_test_0.reshape(-1, 1)).flatten()

        x_train = torch.tensor(x_train_0, dtype=torch.float32)
        y_train = torch.tensor(y_train_0, dtype=torch.float32)

        train_ds = TensorDataset(x_train, y_train)
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        y_vali = torch.tensor(self.y_vali_0, dtype=torch.float32)
        x_vali = torch.tensor(self.x_vali_0, dtype=torch.float32)

        vali_ds = TensorDataset(x_vali, y_vali)
        self.vali_dl = DataLoader(vali_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

def save_scaler(scaler_x,scaler_y):
    mean_x=scaler_x.mean_.tolist()
    var_x=scaler_x.var_.tolist()
    mean_y=scaler_y.mean_.tolist()
    var_y=scaler_y.var_.tolist()

    filename = os.path.join(dirname,'models','current_trial', 'scaler.json')

    with open(filename, 'w') as f:
        json.dump({'mean_x':mean_x, 'var_x':var_x, 'mean_y':mean_y, 'var_y':var_y}, f)

def load_scaler():
    filename = os.path.join(dirname,'models', 'current_trial','scaler.json')
    with open(filename, 'r') as f:
        data = json.load(f)
        mean_x = data['mean_x']
        var_x = data['var_x']
        mean_y = data['mean_y']
        var_y = data['var_y']
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.mean_ = np.array(mean_x)
    scaler_x.var_ = np.array(var_x)
    scaler_y.mean_ = np.array(mean_y)
    scaler_y.var_ = np.array(var_y)
    scaler_x.scale_ = np.sqrt(scaler_x.var_)
    scaler_y.scale_ = np.sqrt(scaler_y.var_)
    return scaler_x, scaler_y




def load_jet_energy_flow_dataset(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None)->JetEnergyCorrectionDataset:
    pts, etas, area, npvs, jec = load_energy_flow(amount=amount, cache_dir=cache_dir, dataset=dataset, subdatasets=subdatasets)
    return JetEnergyCorrectionDataset(pts, etas, area, npvs, jec)