import torch
import math
import pdb
import torch.nn.functional as F

class Conv_Dropout(torch.nn.Module):
    def __init__(self, dropout_rate: int = 0.1, neighbor: int = 2) -> None:
        """
        neighbor: the receptive field in one dimension
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.neighbor = neighbor
    '''
    def get_weight_tensor(self, dim: int) -> torch.Tensor:
        """
        dim(int): input tensor's last dimension
        return: 
            weight_tensor(tensor):weight tensor for one line, each line of this tensor indicate one conv dropout opt
        """
        mu = torch.eye(dim)
        sigma = torch.zeros(dim, dim)
        for n in range(dim):
            sigma[n][max(0, n - self.neighbor): min(dim, n + self.neighbor + 1)] = math.sqrt(self.dropout_rate/(1 - self.dropout_rate)/self.neighbor)
        weight_tensor = torch.normal(mu, sigma)
        return weight_tensor
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) > 1
        #print(input.shape)
        last_dim = input.shape[-1]
        weight_tensor = self.get_weight_tensor(last_dim).to(input.device)
        #pdb.set_trace()
        output = torch.unsqueeze(torch.sum(input.narrow(-2, 0, 1) * weight_tensor, dim=-2), dim=-2)
        for i in range(input.shape[-2] - 1):
            #weight_tensor = self.get_weight_tensor(last_dim).to(input.device)
            one_line = torch.unsqueeze(torch.sum(input.narrow(-2, i + 1, 1) * weight_tensor, dim=-2), dim=-2)
            output = torch.cat([output, one_line], dim=-2)
        #print(output.shape)
        return output
    '''
    def get_bias_tensor(self, biases: torch.Tensor) -> torch.Tensor:
        '''
        biases means x * N(0, sigma^2)
        calculate neighbor(x * N(0, sigma^2))
        '''
        tensor_2D = biases.view(-1, biases.shape[-1]).unsqueeze(1)
        filter = torch.ones(1, 1 , self.neighbor * 2 + 1).to(tensor_2D.device)
        bias = F.conv1d(tensor_2D, filter, padding=self.neighbor)
        return bias.view_as(biases)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        implement as 
        conv_dropout(x) = x + bias
        bias = neighbor(x * N(0, sigma^2))
        '''
        assert len(input.shape) > 1
        weight_tensor = torch.normal(torch.zeros_like(input), torch.ones_like(input) * math.sqrt(self.dropout_rate/(1 - self.dropout_rate)/self.neighbor))
        bias_tensor = self.get_bias_tensor(weight_tensor * input).to(input.device)
        output = bias_tensor + input
        return output
            
    



            

