from collections.abc import Sequence
import torch
from torch import nn
from torchdrug import core, data, utils
from torchdrug.core import Registry as R
from . import layer
from . import utillocal


@R.register("model.WAVBFnet")
class WAVBFnet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, train1_graph, train2_graph, train3_graph, valid_graph, test_graph, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, dependent=True):
        super(WAVBFnet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
       
        train_edge_list1 = train1_graph.edge_list.t().to('cuda')
        train_edge_list1 = train_edge_list1[[1,2,0]]
        train_edge_list2 = train2_graph.edge_list.t().to('cuda')
        train_edge_list2 = train_edge_list2[[1,2,0]]
        train_edge_list3 = train3_graph.edge_list.t().to('cuda')
        train_edge_list3 = train_edge_list3[[1,2,0]]

        #self.eff_wave = utillocal.coefficient_cal(test_graph.adjacency, test_graph.num_node, 'test')
        
        train_exp = torch.tensor(torch.load('/home/InductiveWave/fb_train1.pt')).permute(2,0,1).to('cuda')
        pairs1 = torch.stack([train_edge_list1[2], train_edge_list1[1]], dim=1)
        unique_pairs1, self.inverse1 = torch.unique(pairs1, dim=0, return_inverse=True)
        self.wave_values1 = train_exp[unique_pairs1[:,0], unique_pairs1[:,1]]
        self.wave_values1 = torch.cat((self.wave_values1.real, self.wave_values1.imag), dim=-1).to('cpu')
        
        train_exp = torch.tensor(torch.load('/home/InductiveWave/fb_train2.pt')).permute(2,0,1).to('cuda')
        pairs2 = torch.stack([train_edge_list2[2], train_edge_list2[1]], dim=1)
        unique_pairs2, self.inverse2 = torch.unique(pairs2, dim=0, return_inverse=True)
        self.wave_values2 = train_exp[unique_pairs2[:,0], unique_pairs2[:,1]]
        self.wave_values2 = torch.cat((self.wave_values2.real, self.wave_values2.imag), dim=-1).to('cpu')

        train_exp = torch.tensor(torch.load('/home/InductiveWave/fb_train3.pt')).permute(2,0,1).to('cuda')
        pairs3 = torch.stack([train_edge_list3[2], train_edge_list3[1]], dim=1)
        unique_pairs3, self.inverse3 = torch.unique(pairs3, dim=0, return_inverse=True)
        self.wave_values3 = train_exp[unique_pairs3[:,0], unique_pairs3[:,1]]
        self.wave_values3 = torch.cat((self.wave_values3.real, self.wave_values3.imag), dim=-1).to('cpu')

        
        valid_edge_list = valid_graph.edge_list.t().to('cuda')
        valid_edge_list = valid_edge_list[[1,2,0]]

        train_exp = torch.tensor(torch.load('/home/InductiveWave/fb_valid.pt')).permute(2,0,1).to('cuda')
        pairs_valid = torch.stack([valid_edge_list[2], valid_edge_list[1]], dim=1)
        unique_pairs_valid, self.inverse_valid = torch.unique(pairs_valid, dim=0, return_inverse=True)
        self.wave_values_valid = train_exp[unique_pairs_valid[:,0], unique_pairs_valid[:,1]]
        self.wave_values_valid = torch.cat((self.wave_values_valid.real, self.wave_values_valid.imag), dim=-1)
        
        test_edge_list = test_graph.edge_list.t().to('cuda')
        test_edge_list = test_edge_list[[1,2,0]]
        train_exp = torch.tensor(torch.load('/home/InductiveWave/fb_test.pt')).permute(2,0,1).to('cuda')
        pairs_test = torch.stack([test_edge_list[2], test_edge_list[1]], dim=1)
        unique_pairs_test, self.inverse_test = torch.unique(pairs_test, dim=0, return_inverse=True)
        self.wave_values_test = train_exp[unique_pairs_test[:,0], unique_pairs_test[:,1]]
        self.wave_values_test = torch.cat((self.wave_values_test.real, self.wave_values_test.imag), dim=-1)
        
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))
    
    def forward(self, graph, edge_mask, input, split, all_loss=None, metric=None, eph=0):
        
        with graph.node():
            graph.boundary = input
        hiddens = []
        layer_input = input
        if split=='train':
            if eph==0:
                self.wave_values = self.wave_values1
                self.inverse = self.inverse1
            elif eph==1:
                self.wave_values = self.wave_values2
                self.inverse = self.inverse2
            else:
                self.wave_values = self.wave_values3
                self.inverse = self.inverse3
                    
            self.wave_values = self.wave_values.cuda()
            self.wave_values_valid = self.wave_values_valid.cpu()
            self.wave_values_test = self.wave_values_test.cpu()
        
            self.inverse = self.inverse[edge_mask]

        torch.cuda.empty_cache()
        for layer in self.layers:
            if split=='valid':

                if not(self.wave_values_valid.is_cuda):
                    self.wave_values_valid = self.wave_values_valid.cuda()
                    self.wave_values = self.wave_values.cpu()
                    self.wave_values_test = self.wave_values_test.cpu()
                
                hidden = layer(graph, layer_input, self.inverse_valid, self.wave_values_valid)
            elif split=='test':

                if not(self.wave_values_test.is_cuda):
                    self.wave_values_test = self.wave_values_test.cuda()
                    self.wave_values_valid = self.wave_values_valid.cpu()
                    self.wave_values = self.wave_values.cpu()
                
                hidden = layer(graph, layer_input, self.inverse_test, self.wave_values_test)
            else:
                hidden = layer(graph, layer_input, self.inverse, self.wave_values)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = graph.query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": node_feature,
        }

