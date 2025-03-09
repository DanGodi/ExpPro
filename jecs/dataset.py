import energyflow as ef
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load(amount=0.1, cache_dir='~/.energyflow', dataset='sim', subdatasets=None):
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

    x=np.transpose(np.array([pts,etas,area,npvs,gen_pts]))
    y=np.array(jec)

    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(x, y, test_size=0.2, shuffle=True)
    x_train_0,x_vali_0,y_train_0,y_vali_0 = train_test_split(x_train_0 , y_train_0 , test_size=0.4, shuffle=True)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_0 = scaler_x.fit_transform(x_train_0)
    x_vali_0 = scaler_x.transform(x_vali_0)
    x_test_0 = scaler_x.transform(x_test_0)

    y_train_0 = scaler_y.fit_transform(y_train_0.reshape(-1, 1)).flatten()
    y_vali_0 = scaler_y.transform(y_vali_0.reshape(-1, 1)).flatten()
    y_test_0 = scaler_y.transform(y_test_0.reshape(-1, 1)).flatten()

    x_train = torch.tensor(x_train_0[:, :-1], dtype=torch.float32)
    y_train = torch.tensor(y_train_0, dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

    y_vali = torch.tensor(y_vali_0, dtype=torch.float32)
    x_vali = torch.tensor(x_vali_0[:, :-1], dtype=torch.float32)

    vali_ds = TensorDataset(x_vali, y_vali)
    vali_dl = DataLoader(vali_ds, batch_size=512, shuffle=True)

    return x_vali_0, y_vali_0, x_test_0, y_test_0, train_dl, vali_dl, scaler_x, scaler_y