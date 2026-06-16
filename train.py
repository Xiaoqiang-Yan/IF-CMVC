import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from network import Network,MultiHeadAttention, FeedForwardNetwork
from metric import valid
from hsic import hsic_normalized_cca 
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss 
from dataloader import load_data
import os
import datetime
from lightly.loss import NTXentLoss
from utils import UD_constraint, unite, EarlyStopping
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Event8
# Handwritten
# Flickr
# CCV
# BDGP
# Caltech-2V
# Caltech-3V
# Caltech-4V
# COCO
Dataname = 'Caltech-2V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--sig", default=1.0) 
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pretrain_epoc", default=200)  
parser.add_argument("--train_epoc", default=300)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--alpha", default=0.0001)
parser.add_argument("--beta", default=0.1)
parser.add_argument("--gamma", default=0.01)
parser.add_argument("--seed", type=int, default=6)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, zs, _, _, _,_= model(xs=xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss = tot_loss + loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def train(epoch, p_sample, adaptive_weight,losses):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    NTX_loss = NTXentLoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):       
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, x_layers, x_hs, x_qs,x_de_layers= model(xs=xs)           
        loss_list = [] 
        loss_DPI=[]
        loss_CL=[]
        
        for v in range(view):       
            for w in range(v + 1, view):                              
                loss_list.append(NTX_loss(hs[v], hs[w]))               
                loss_list.append(NTX_loss(qs[v], qs[w]))              
                for i in range(len(x_layers[0])-1): 
                    loss_CL.append(NTX_loss(x_hs[v][i], x_hs[w][i])+NTX_loss(x_qs[v][i], x_qs[w][i])) 
            loss_list.append(mes(xs[v], xrs[v]))
            UDC = UD_constraint(qs[v]).to(device)
            loss_list.append(cross_entropy(qs[v], UDC))      
            for i in range(len(x_layers[0])):                          
                x_layer = x_layers[v][i]                           
                x_h = x_hs[v][i]                                   
                x_q = x_qs[v][i]  
                x_de_layer=x_de_layers[v][len(x_layers[0])-i-1]   
                
                xs_data = xs[v].view(-1, np.prod(xs[v].size()[1:]))    
                layer_data = x_layer.view(-1, np.prod(x_layer.size()[1:]))  
                hs_data = x_h.view(-1, np.prod(x_h.size()[1:]))      
                qs_data = x_q.view(-1, np.prod(x_q.size()[1:]))     
                layer_de_data = x_de_layer.view(-1, np.prod(x_de_layer.size()[1:])) 
        
                hsic_xs_zs = hsic_normalized_cca(xs_data, layer_data,args.sig, device)
                hsic_hs_qs = -hsic_normalized_cca(hs_data, qs_data,args.sig, device)
                hsic_zs_de = -hsic_normalized_cca(layer_data,layer_de_data,args.sig, device)
                loss_list.append(args.gamma*hsic_hs_qs)  
                loss_DPI.append(hsic_xs_zs+hsic_zs_de)
                               
        feature_zs = unite(view, zs, attention_net, p_net, p_sample, adaptive_weight)   
        feature_hs = unite(view, hs, attention_net, p_net, p_sample, adaptive_weight)
        feature_qs = unite(view, qs, attention_net, p_net, p_sample, adaptive_weight)
        hsic_fea_hs_qs = -hsic_normalized_cca(feature_hs, feature_qs, args.sig)
        for v in range(view):
            xs_da = xs[v].view(-1, np.prod(xs[v].size()[1:])) 
            hsic_fea_xs_zs = hsic_normalized_cca(xs_da, feature_zs, args.sig, device)
            loss_list.append(args.gamma*(hsic_fea_xs_zs+hsic_fea_hs_qs))
            
        loss = sum(loss_list)+args.alpha*sum(loss_DPI)+args.beta*sum(loss_CL)     #total loss
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    losses.append(tot_loss / len(data_loader))
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)
     
        
accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))
    setup_seed(args.seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    attention_net = MultiHeadAttention(args.hidden_dim, args.attention_dropout_rate, args.num_heads,args.attn_bias_dim)  
    attention_net = attention_net.to(device)
    p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate) 
    p_net = p_net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
   
    # init p distribution  
    p_sample = np.ones(view)  
    p_sample = p_sample / sum(p_sample)  
    p_sample = torch.FloatTensor(p_sample).cuda()

    # init adaptive weight  
    adaptive_weight = np.ones(view)
    adaptive_weight = adaptive_weight / sum(adaptive_weight)
    adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()  
    adaptive_weight = adaptive_weight.unsqueeze(1)  
                                                                  
    losses = []
    epoch = 1
    early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-1)
    while epoch <= args.pretrain_epoc:  
        pretrain(epoch)
        epoch += 1
    while epoch <= args.pretrain_epoc + args.train_epoc:
        model.train()
        loss_ave=train(epoch, p_sample, adaptive_weight,losses)
        if early_stopper.step(loss_ave):            
            print(f"Early stopping triggered at epoch {epoch}")
            acc, nmi, pur = valid(model, device, dataset, view, data_size)
            break
        if epoch % 10 == 0:                            
            acc, nmi, pur = valid(model, device, dataset, view, data_size)
        epoch += 1
  
        
        
        
        


    