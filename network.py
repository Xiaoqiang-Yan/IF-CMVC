import torch.nn as nn
from torch.nn.functional import normalize
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.layer1=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 500),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(500, 2000),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2000, feature_dim)
            # nn.ReLU()
        )
        
    def forward(self, x):
        out_layers=[]
        out=self.layer1(x)
        out_layers.append(out)
        out=self.layer2(out)
        out_layers.append(out)
        out=self.layer3(out)
        out_layers.append(out)
        out=self.layer4(out)
        out_layers.append(out)
        return out,out_layers


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, 2000),
            nn.ReLU()
        )  
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2000, 500),
            nn.ReLU()
        )  
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(500, 500),
            nn.ReLU()
        )  
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(500, input_dim)
            # nn.ReLU()
        )  
    def forward(self, x):
        de_layers = []
        de_layers.append(x)
        out = self.layer1(x)
        de_layers.append(out)
        out = self.layer2(out)
        de_layers.append(out)
        out = self.layer3(out)
        de_layers.append(out)
        out = self.layer4(out)
        return out,de_layers


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):          
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module1= nn.Sequential(          
            nn.Linear(500,500),                
            nn.ReLU(),
            nn.Linear(500, high_feature_dim)
        )
        
        self.feature_contrastive_module2= nn.Sequential(          
            nn.Linear(2000,2000),               
            nn.ReLU(),
            nn.Linear(2000, high_feature_dim)
        )
         
        self.feature_contrastive_module3= nn.Sequential(        
            nn.Linear(feature_dim, feature_dim),               
            nn.ReLU(),
            nn.Linear(feature_dim, high_feature_dim)
        )
            
        self.label_contrastive_module1= nn.Sequential(
            nn.Linear(500, class_num),        
            nn.Softmax(dim=1)
        )
        self.label_contrastive_module2= nn.Sequential(
            nn.Linear(2000, class_num),       
            nn.Softmax(dim=1)
        )
        self.label_contrastive_module3= nn.Sequential(
            nn.Linear(feature_dim, class_num),       
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        c_layers=[]                            
        c_hs=[]                                
        c_qs=[]                                
        c_de_layers=[]                         
        for v in range(self.view):
            x = xs[v]
            z,c_layer = self.encoders[v](x)                                
            h1 = normalize(self.feature_contrastive_module3(z), dim=1)
            q1 = self.label_contrastive_module3(z)
            c_h=[]
            c_q=[]
            for r in range(len(c_layer)):
                if r==0 or r==1:
                    h = normalize(self.feature_contrastive_module1(c_layer[r]), dim=1)
                    c_h.append(h)
                elif r==2:
                    h = normalize(self.feature_contrastive_module2(c_layer[r]), dim=1)
                    c_h.append(h)
                else:
                    h = normalize(self.feature_contrastive_module3(c_layer[r]), dim=1)
                    c_h.append(h)
                    
            for r in range(len(c_layer)):
                if r==0 or r==1:
                    q = self.label_contrastive_module1(c_layer[r])
                    c_q.append(q)
                elif r==2:
                    q = self.label_contrastive_module2(c_layer[r])
                    c_q.append(q)
                else:
                    q = self.label_contrastive_module3(c_layer[r])
                    c_q.append(q)
            # xr = self.decoders[v](z)
            xr,c_de_layer=self.decoders[v](z)
            hs.append(h1)
            zs.append(z)
            qs.append(q1)
            xrs.append(xr)
            c_layers.append(c_layer)               
            c_de_layers.append(c_de_layer)
            c_hs.append(c_h)                        
            c_qs.append(c_q)                       
        return hs,qs, xrs, zs,c_layers,c_hs,c_qs,c_de_layers

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module3(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z,_= self.encoders[v](x)           
            q = self.label_contrastive_module3(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):  
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads                                    

        self.att_size = att_size = hidden_size // num_heads           
        self.scale = att_size ** -0.5                                  

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)   
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        q = q * self.scale
        x = torch.matmul(q, k)                 # [b, h, q_len, k_len]                   

        x = torch.softmax(x, dim=3)                                      
        x = self.att_dropout(x)
        x = x.matmul(v)                        # [b, h, q_len, attn]                           

        x = x.transpose(1, 2).contiguous()     # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)

        x = self.output_layer(x)

        return x

    
class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(view, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x
