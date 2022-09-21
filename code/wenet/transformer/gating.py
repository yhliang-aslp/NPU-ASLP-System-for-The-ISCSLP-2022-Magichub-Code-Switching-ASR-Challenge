import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

class Gate(torch.nn.Module):
    def __init__(self,attention_dim=256):
        super(Gate, self).__init__()
        self.linear_all = torch.nn.Linear(attention_dim, 2)


    def forward(self,x_cn,x_en):
        x_cn = self.linear_all(x_cn)
        x_en = self.linear_all(x_en)
        x = x_cn + x_en
        x = torch.softmax(x, dim=-1)

        # fig0 = plt.figure()
        # len = x.shape[1]
        # plt.plot([z for z in range(0, len)], x[0,:,1].cpu().detach().numpy(),label = "en")
        # plt.ylim((0,1))
        # fig0.savefig("05.png")
        # exit()

        # print(x)
        # pdb.set_trace()
        return x


class Gate_fsmn(torch.nn.Module):
    def __init__(self,attention_dim = 256, class_num = 2, l_order = 15, r_order = 15, ffn_inner_dim = 64, dropout=0.1):
        super(Gate_fsmn, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(attention_dim, ffn_inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ffn_inner_dim),
            nn.Linear(ffn_inner_dim, class_num,bias=False))
        self.dropout = nn.Dropout(p=dropout)
        
        self.pad=nn.ConstantPad1d((l_order,r_order),0)
        self.depthwise_conv2d = nn.Conv1d(class_num,class_num,l_order+r_order+1,groups=1)
    def dfsmn(self, padded_inputs):
        context = self.fc(padded_inputs)
        context = self.dropout(context)
        context = context.transpose(1,2)
        queries = self.pad(context)
        memory = self.depthwise_conv2d(queries)
        output = context +memory
        output = self.dropout(output)
        output = output.transpose(1,2)
        return output
    def forward(self,x_cn,x_en):
        x_en = self.dfsmn(x_en)
        x_cn = self.dfsmn(x_cn)
        x = x_cn + x_en
        x = torch.softmax(x, dim=-1)
        #pdb.set_trace()
        print("gatting DFSMN")
        return x

class Gate_linear_cnn(torch.nn.Module):
    def __init__(self,attention_dim = 256, class_num = 2, l_order = 5, r_order = 5, ffn_inner_dim = 64, dropout=0.1):
        super(Gate_linear_cnn, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(attention_dim, class_num),
            nn.ReLU(),
            nn.LayerNorm(class_num))
        self.pad=nn.ConstantPad1d((l_order,r_order),0)
        self.depthwise_conv2d = nn.Conv1d(class_num,class_num,l_order+r_order+1,groups=1)
    def dfsmn(self, padded_inputs):
        context = self.fc(padded_inputs)
        context = context.transpose(1,2)
        queries = self.pad(context)
        memory = self.depthwise_conv2d(queries)
        output = memory.transpose(1,2)
        return output
    def forward(self,x_cn,x_en):
        x_en = self.dfsmn(x_en)
        x_cn = self.dfsmn(x_cn)
        x = x_cn + x_en
        x = torch.softmax(x, dim=-1)
        #pdb.set_trace()
        print("Gate_linear_cnn")
        return x
class Gate_cnn_linear(torch.nn.Module):
    def __init__(self,attention_dim = 256, class_num = 2, l_order = 5, r_order = 5, ffn_inner_dim = 64, dropout=0.1):
        super(Gate_cnn_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(attention_dim, class_num),
            nn.ReLU(),
            nn.LayerNorm(class_num))        
        self.pad=nn.ConstantPad1d((l_order,r_order),0)
        self.depthwise_conv2d = nn.Conv1d(attention_dim,attention_dim,l_order+r_order+1,groups=1)
    def dfsmn(self, padded_inputs):
        context = padded_inputs.transpose(1,2)
        queries = self.pad(context)
        memory = self.depthwise_conv2d(queries).transpose(1,2)
        output = self.fc(memory)
        return output
    def forward(self,x_cn,x_en):
        x_en = self.dfsmn(x_en)
        x_cn = self.dfsmn(x_cn)
        x = x_cn + x_en
        x = torch.softmax(x, dim=-1)
        #pdb.set_trace()
        print("Gate_linear_cnn")
        return x
class Gate_cnn_linear_no_relu_no_laynorm(torch.nn.Module):
    def __init__(self,attention_dim = 256, class_num = 2, l_order = 5, r_order = 5):
        super(Gate_cnn_linear_no_relu_no_laynorm, self).__init__()
        self.fc = nn.Linear(attention_dim, class_num)    
        self.pad=nn.ConstantPad1d((l_order,r_order),0)
        self.depthwise_conv2d = nn.Conv1d(attention_dim,attention_dim,l_order+r_order+1,groups=attention_dim)
    def dfsmn(self, padded_inputs):
        context = padded_inputs.transpose(1,2)
        queries = self.pad(context)
        memory = self.depthwise_conv2d(queries).transpose(1,2)
        output = self.fc(memory)
        return output
    def forward(self,x_cn,x_en):
        x_en = self.dfsmn(x_en)
        x_cn = self.dfsmn(x_cn)
        x = x_cn + x_en
        x = torch.softmax(x, dim=-1)
        #pdb.set_trace()
        print("Gate_cnn_linear_no_relu_no_laynorm")
        return x
if __name__ == "__main__":
    x1 = torch.randn(2,3,5)
    x2 = torch.randn(2,3,5)
    G = Gate(attention_dim=5)
    y = G(x1,x2)
    print(y)
    print(x1)
    print(y[:,:,0])
    q1 = x1*(y[:,:,0].unsqueeze(dim=2))
    q2 = x2*(y[:,:,1].unsqueeze(dim=2))
    print(q1)