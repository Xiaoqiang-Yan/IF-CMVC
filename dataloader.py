from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

class Event(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'event.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'event.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path+'event.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'event.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]                        

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]),torch.from_numpy(self.x3[idx])],torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Hand(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'handwritten.mat')
        X_data = mat['X']
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(X_data[0, 0]).astype(np.float32)
        self.V2 = scaler.fit_transform(X_data[0, 1]).astype(np.float32)
        self.V3 = scaler.fit_transform(X_data[0, 2]).astype(np.float32)
        self.V4 = scaler.fit_transform(X_data[0, 3]).astype(np.float32)
        self.V5 = scaler.fit_transform(X_data[0, 4]).astype(np.float32)
        self.V6 = scaler.fit_transform(X_data[0, 5]).astype(np.float32)
        self.Y = np.array(np.squeeze(mat['Y'])).astype(np.int32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx]), torch.from_numpy(self.V6[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    
class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
    
    
class Flickr(Dataset):
    def __init__(self, path):      
        self.Y = scipy.io.loadmat(path + 'flickr.mat')['Y'].astype(np.int32).reshape(12154, )
        self.V1 = scipy.io.loadmat(path + 'flickr.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'flickr.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'flickr.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'flickr.mat')['X4'].astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1).astype(np.float32)
        self.V2 = scaler.fit_transform(self.V2).astype(np.float32)
        self.V3 = scaler.fit_transform(self.V3).astype(np.float32)
        self.V4 = scaler.fit_transform(self.V4).astype(np.float32)

    def __len__(self):     
        return 12154

    def __getitem__(self, idx):    

        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "Event8":
        dataset =Event('./data/')
        dims = [1000,1000,1000]
        view = 3
        data_size = 1579
        class_num = 8
    elif dataset == "Handwritten":
        dataset = Hand('./data/')
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "Flickr":
        dataset = Flickr('data/')
        dims = [4096, 1000,768,100]
        view = 4
        data_size = 12154
        class_num = 6
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
