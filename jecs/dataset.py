import energyflow as ef
import numpy as np
import torch
import uproot
import awkward as ak
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset



def load_energy_flow(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None):
    pts = []
    etas = []
    gen_pts = []
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
        gen_pts.append(data.gen_jet_pts[j])
        jec.append(data.jecs[j])
    
    return pts, etas, area, npvs, gen_pts, jec

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
        gen_pts = np.zeros(len(pts)) # zero for now later maybe we do matching
        jec = np.hstack(jets['jet_jec'])
        yield pts, etas, areas, npvs, gen_pts, jec
        


class JetEnergyCorrectionDataset(TensorDataset):
    def __init__(self, pts, etas, area, npvs, gen_pts, jec):
        self.__load(pts, etas, area, npvs, gen_pts, jec)

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
        

    def __load(self,pts, etas, area, npvs,gen_pts, jec):
        x=np.transpose(np.array([pts,etas,area,npvs,gen_pts]))
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

        x_train = torch.tensor(x_train_0[:, :-1], dtype=torch.float32)
        y_train = torch.tensor(y_train_0, dtype=torch.float32)

        train_ds = TensorDataset(x_train, y_train)
        self.train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

        y_vali = torch.tensor(self.y_vali_0, dtype=torch.float32)
        x_vali = torch.tensor(self.x_vali_0[:, :-1], dtype=torch.float32)

        vali_ds = TensorDataset(x_vali, y_vali)
        self.vali_dl = DataLoader(vali_ds, batch_size=512, shuffle=True)


def load_jet_energy_flow_dataset(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None)->JetEnergyCorrectionDataset:
    pts, etas, area, npvs, gen_pts, jec = load_energy_flow(amount=amount, cache_dir=cache_dir, dataset=dataset, subdatasets=subdatasets)
    return JetEnergyCorrectionDataset(pts, etas, area, npvs, gen_pts, jec)