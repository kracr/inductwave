import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import layers, utils
from torchdrug.layers import functional
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm((output_dim//2), eps=1e-3)#.to(torch.float32)
            self.layer_norm2 = nn.LayerNorm(output_dim//2, eps=1e-3)#.to(torch.float32)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.w11 = nn.Parameter(torch.ones(32)).to(dtype=torch.half, device='cuda')
        self.w21 = nn.Parameter(torch.ones(32)).to(dtype=torch.half, device='cuda')
        if self.aggregate_func == "pna":
            self.linear = nn.Linear((output_dim//2)*14, (output_dim//2))
            self.linear2 = nn.Linear((output_dim//2)*14, output_dim//2)
        else:
            self.linear = nn.Linear((input_dim//2) * 3, (output_dim//2)).float()#.half()
            self.linear2 = nn.Linear((input_dim//2) * 3, (output_dim//2)).float()#.half()

        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, (num_relation) * input_dim)#.float()#.half()
        else:
            self.relation = nn.Embedding(num_relation, input_dim)
            self.relation.weight.data.uniform_(-1, 1)

    def forward(self, graph, input, wave_indices, wave_values):
        message1, message2 = self.message(graph, input, wave_indices, wave_values)
        
        output = self.combine(input, message1, message2)
        return output


    def message(self, graph, input, wave_indices, wave_values):
        assert graph.num_relation == self.num_relation
        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary#.flatten(1)
        degree_out = graph.degree_out[:, None, None]+1

        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        
        factor = wave_values.index_select(0, wave_indices)
        factor.mul_(self.w21).add_(self.w11)
        size_a, size_b, size_c = graph.adjacency.size()
        wave_val = torch.sparse_coo_tensor(graph.edge_list.t()[[1,0,2]], factor, size=(size_a, size_b, size_c, 32))
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]

        #if not pna comment the other aggregators
        mean = functional.generalized_rspmm(wave_val, relation_input, input, sum="add", mul=mul)
        sq_mean = functional.generalized_rspmm(wave_val ** 2, relation_input ** 2, input ** 2, sum="add", mul=mul)
        max = functional.generalized_rspmm(wave_val, relation_input, input, sum="max", mul=mul)
        min = functional.generalized_rspmm(wave_val, relation_input, input, sum="min", mul=mul)
        
        mean=mean.view(len(mean), batch_size, -1)
        max=max.view(len(max), batch_size, -1)
        min=min.view(len(min), batch_size, -1)
        sq_mean=sq_mean.view(len(sq_mean), batch_size, -1)
        
        mean = (mean + boundary) / degree_out
        sq_mean = (sq_mean + boundary ** 2) / degree_out
        max = torch.max(max, boundary)
        min = torch.min(min, boundary)
        std = torch.addcmul(sq_mean, mean, mean, value=-1.0).clamp(min=self.eps).sqrt()
        
        mean1, mean2 = mean.chunk(2, dim=-1)
        max1, max2 = max.chunk(2, dim=-1)
        min1, min2 = min.chunk(2, dim=-1)
        std1, std2 = std.chunk(2, dim=-1)
        
        features1 = torch.cat([mean1, max1, min1, std1, mean2, max2, min2, std2], dim=-1)#.flatten(-2)
        scale1 = degree_out.log()
        scale1 = scale1 / scale1.mean()
        inv_scale1 = torch.reciprocal(scale1.clamp_min_(1e-2))
        scales1 = torch.cat([torch.ones_like(scale1), scale1, inv_scale1], dim=-1)
        update_arr1 = (features1[..., None] * scales1[:, :, None, :]).reshape(*features1.shape[:-1], -1)
        
        return update_arr1, None 

    def combine(self, input, update1, update2):
        split1, split2 = update1.chunk(2, dim=-1)
        output1 = self.linear(torch.cat([input, split1], dim=-1))
        output2 = self.linear2(torch.cat([input, split2], dim=-1))
        if self.layer_norm:
            output1 = self.layer_norm(output1)
            output2 = self.layer_norm2(output2)
        
        output = torch.cat([output1, output2], dim=-1)
        if self.activation:
            output = self.activation(output)

        return output

