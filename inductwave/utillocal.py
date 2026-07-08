import torch
import pickle
import cupyx, cupy
import scipy
from tqdm import tqdm
import numpy as np


pi = torch.acos(torch.zeros(1)).item() * 2

class Filter(object):

    def __init__(self, kernels):

        try:
            iter(kernels)
        except TypeError:
            kernels = [kernels]
        self._kernels = kernels


class Ricker(Filter):

    def __init__(self, lmax_all, tau=10,  **kwargs):

        def low_pass(x, t):
            return np.exp(-t * x)

        def rick(x,t):
            return low_pass(x,t)
        
        g = lambda x, t=tau, scales = scales : rick(x, scales, t)

        super(Ricker, self).__init__(g, **kwargs)


def compute_cheby_coeff(f, lmax_all,  m=30, N=None, *args, **kwargs):
    i = kwargs.pop('i', 0)

    if not N:
        N = m + 1

    a_arange = [0, lmax_all]

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.
    
    c = np.zeros(m+1)

    tmpN = np.arange(N)
    num = np.cos(pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * np.dot(f._kernels[i](a1 * num + a2),
                               np.cos(np.pi * o * (tmpN + 0.5) / N))
    c = np.array([c])
    return c


def cheby_op4(num_node, c, signal, laplacian_first, laplac2, lmax_all, a1, a2, factor2):
        _, M = c.shape        # M - number of cheby approximation+1
        
        twf_old2 = np.repeat(signal, laplac2.shape[0], axis=0)
        twf_cur2 = np.array([((laplac2[i].dot(signal) - a2* twf_old2[i]) / a1) for i in range(laplac2.shape[0])])

        r_imag = np.array([0.5 * c[0, 0] * twf_old2[i] + c[0, 1] * twf_cur2[i] for i in range(twf_cur2.shape[0])])
        
        for k in tqdm(range(2, M)):
            twf_new2 = np.array([factor2[i].dot(twf_cur2[i]) - twf_old2[i] for i in range(laplac2.shape[0])])
            r_imag += c[0, k] * twf_new2
            
            twf_old2 = twf_cur2
            twf_cur2 = twf_new2 

        return r_imag


def cheby_op3(num_node, c, signal, laplacian_first, laplac2, lmax_all):
        Nscales, M = c.shape        # M - number of cheby approximation+1

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")
        
        a_arange = [0, lmax_all]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.
        
        eye = scipy.sparse.eye(num_node, dtype=np.float32)
        twf_old1 = signal

        twf_cur1 = (laplacian_first - a2 * twf_old1) / a1
        
        r_real = 0.5 * c[0, 0] * twf_old1 + c[0, 1] * twf_cur1
        
        factor1 = 2/a1 * (laplacian_first - (a2 * eye))
        
        for k in tqdm(range(2, M)):
            twf_new1 = factor1.dot(twf_cur1) - twf_old1
            r_real += c[0, k] * twf_new1
            
            twf_old1 = twf_cur1
            twf_cur1 = twf_new1
        return r_real

def coefficient_cal(train_adj, train_num_nodes, split):
        path = '/home/InductiveWave/' 
        num_node = train_num_nodes
        adj = train_adj
        adj = adj.permute(2,0,1)
        laplacian = laplacian_cal(adj)
        
        lmax_all = 2
        filters = Ricker(lmax_all = lmax_all)
        chebyshev = compute_cheby_coeff(filters, lmax_all, m=37)
        
        laplac2 = None
        for j in tqdm(range(laplacian.size()[0])):
            val = laplacian[j].coalesce().values()
            row = laplacian[j].coalesce().indices()[0]
            col = laplacian[j].coalesce().indices()[1]
            laplacian_sec = scipy.sparse.coo_matrix((val.numpy(),(row.numpy(), col.numpy())), shape = laplacian[j].size())
            
            if laplac2 is None:
                laplac2 = np.array([laplacian_sec])
            else:
                laplac2 = np.concatenate((laplac2, np.array([laplacian_sec])), axis=0)
        
        
        eff_wave_arr = None
        
        j=0
        
        laplac2 = None
        
        for j in tqdm(large_relation):
            val = laplacian[j].coalesce().values()
            threshold=1e-6
            mask1= val.abs() > threshold
            val = val[mask1]
            row = laplacian[j].coalesce().indices()[:,mask1][0]
            col = laplacian[j].coalesce().indices()[:,mask1][1]
            laplacian_sec = scipy.sparse.coo_matrix((val.numpy(),(row.numpy(), col.numpy())), shape = laplacian[j].size())
            
            if laplac2 is None:
                laplac2 = np.array([laplacian_sec])
            else:
                laplac2 = np.concatenate((laplac2, np.array([laplacian_sec])), axis=0)
        siz=num_node
        '''
        Following code is for calculating wavelets one relation at a time uncomment it if needed
        for i in tqdm(range(0, laplacian.size()[0])):
            laplacian_first = None
            
            if i in large_relation:
                continue
                
            val = laplacian[i].coalesce().values()
            threshold=1e-6
            mask1= val.abs() > threshold
            val = val[mask1]
            row = laplacian[i].coalesce().indices()[:,mask1][0]
            col = laplacian[i].coalesce().indices()[:,mask1][1]
            laplac2 = scipy.sparse.coo_matrix((val.numpy(),(row.numpy(), col.numpy())), shape = laplacian[i].size())
                
            impulse = scipy.sparse.coo_matrix((np.ones(siz),(np.array([j for j in range(siz)]), np.arange(siz))), shape = (num_node, siz), dtype=np.float32)
                
            im = cheby_op3(num_node.numpy(), chebyshev, impulse, laplac2, laplacian_first, lmax_all)
            
            eff_wave = im
            
            if eff_wave_arr is None:
                eff_wave_arr = np.array([eff_wave])
            else:
                eff_wave_arr = np.concatenate((eff_wave_arr, np.array([eff_wave])), axis=0)
            torch.save(eff_wave_arr, f'/home/InductWave/wiki_coeff/wave_{i}.pt', pickle_protocol=5)
            eff_wave_arr=None
                    
        exit()
        '''
        siz = num_node
        j=0
        
        a_arange = [0, lmax_all]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.
    
        eye = scipy.sparse.identity(num_node)
        factor2 = np.array([2/a1 * (laplac2[i] - a2 * eye) for i in range(laplac2.shape[0])])
        
        for i in tqdm(range((0)*(siz), int(num_node), siz)):
            siz = min(siz, int(num_node)-i)
            
            impulse = scipy.sparse.coo_matrix((np.ones(siz),(np.array([i+j for j in range(siz)]), np.arange(siz))), shape = (num_node, siz))
            
            laplacian_first = None
            im = cheby_op4(num_node.numpy(), chebyshev, impulse, laplacian_first, laplac2, lmax_all, a1, a2, factor2)
            if eff_wave_arr is None:
                eff_wave_arr = np.array([im])
            else:
                eff_wave_arr = np.concatenate((eff_wave_arr, np.array([im])), axis=0)
        eff_wave_raw = post(eff_wave_arr)
        real_imag = approximate_wavelet_embedding(eff_wave_raw)
        torch.save(real_imag, f'{path}/fb_{split}.pt', pickle_protocol=5)
        return 0

def approximate_wavelet_embedding(eff_wave_raw):
    step_size1 = 4
    step_size2 = 3
    sample_number = 16
    steps1 = [x*step_size1 for x in range(sample_number)]
    steps2 = [x*step_size2 for x in range(sample_number)]

    real_imag_arr = None
    for i in tqdm(range(eff_wave_raw.size()[0])):
        eff_wave_coalesced = eff_wave_raw[i].coalesce()
        val1 = eff_wave_coalesced.values().real.cpu()
        val2 = eff_wave_coalesced.values().imag.cpu()
        ind = eff_wave_coalesced.indices().cpu()
        raw1 = cupyx.scipy.sparse.coo_matrix((cupy.array(val1.numpy()),(cupy.array(ind[0].numpy()), cupy.array(ind[1].numpy()))), shape = eff_wave_raw[i].size())
        raw2 = cupyx.scipy.sparse.coo_matrix((cupy.array(val2.numpy()),(cupy.array(ind[0].numpy()), cupy.array(ind[1].numpy()))), shape = eff_wave_raw[i].size())


        real_imag = np.array([cupy.squeeze(cupyx.scipy.sparse.csr_matrix.mean(cupyx.scipy.sparse.csr_matrix.expm1(((raw1*1*step1)+(raw2*1*step2))*1j).astype(np.complex64), axis=0)+1).get() for step1,step2 in zip(steps1,steps2)])
        if real_imag_arr is None:
            real_imag_arr = real_imag[np.newaxis, :]
        else:
            real_imag_arr = np.concatenate((real_imag_arr,real_imag[np.newaxis, :]), axis = 0)
    return real_imag_arr

def post(x):
    wave_arr = None
    for i in tqdm(range(len(x[0]))):
        row, col = x[0][i].nonzero()
        ind = torch.tensor([row, col])
        val = x[0][i].data
        shape = torch.Size(x[0][i].shape)
        wave_mat = torch.sparse_coo_tensor(ind, val, shape)
        if wave_arr is None:
            wave_arr = wave_mat.unsqueeze(0)
        else:
            wave_arr = torch.cat((wave_arr, wave_mat.unsqueeze(0)), dim=0)

    return wave_arr


def laplacian_cal(adj):
        
        q = 0.25
        
        A_s = 0.5*(adj+torch.transpose(adj, -2, -1)).to(dtype=torch.float32)
        flat_adj = torch.sum(A_s,0)
        
        diag = torch.sparse.sum(flat_adj,dim=1).to_dense().to(torch.float32)
        diag_ind = torch.stack([torch.sparse.sum(A_s[i], dim=1).to_dense().to(torch.float32) for i in range(A_s.size(0))])
        
        diag_updated = torch.stack([torch.sparse.spdiags(diag_ind[i], torch.tensor([0]),(len(diag_ind[i]), len(diag_ind[i]))) for i in range(diag_ind.size(0))])
        diag = torch.where(diag==0, diag, torch.pow(diag, -0.5))
        diag_updated2 = torch.sparse.spdiags(diag, torch.tensor([0]),(len(diag), len(diag))).to(dtype=torch.cfloat)
       
        theta = 2*pi*q*1j*(adj-torch.transpose(adj, -2, -1)).to(dtype=torch.cfloat)
        
        value_theta = theta.coalesce().values()
        index_theta = theta.coalesce().indices()

        theta = torch.sparse_coo_tensor(index_theta, torch.exp(value_theta), size=theta.size())
        
        A_s = torch.stack([A_s[i]*theta[i] for i in range(A_s.size(0))]).to_sparse().coalesce()
        lap_unnorm = torch.stack([diag_updated[i]-A_s[i] for i in range(A_s.size(0))])
        norm_mat=None
        norm_mat = torch.stack([torch.sparse.mm(diag_updated2, torch.sparse.mm(lap_unnorm_i, diag_updated2)) for lap_unnorm_i in lap_unnorm]).to_sparse()
        return norm_mat
