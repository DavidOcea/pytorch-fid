#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import models
import torch.nn.functional as F
import os

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename).resize((112,112)), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=1, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if batch_size > len(files):
    #     print(('Warning: batch size is bigger than the data size. '
    #            'Setting batch size to data size'))
    #     batch_size = len(files)
    batch_size = 1
    pred_arr = np.empty((len(files), dims))

    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
        
        # Reshape to (n_images, 3, height, width)
        
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if len(pred.shape)>2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')
    if len(pred.shape) == 4:
        pred = torch.flatten(pred, start_dim=1, end_dim=-1)
    else:
        # pred = pred.unsqueeze(0) #不normalize
        pred = F.normalize(pred.unsqueeze(0),p=2,dim=1) #L2_normalize

    return pred_arr, pred


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act, apr = get_activations(files, model, batch_size, dims, cuda, verbose) #1x2048-np;1x2048x1x1-tensor 后1X2048
    act = np.array(act,dtype=np.float64)
    mu = np.mean(act, axis=0) #2048
    sigma = np.cov(act, rowvar=False) #一个值 协方差
    return mu, sigma, apr


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = [path]
        m, s, a = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s, a


def calculate_fid_given_paths(paths, batch_size, cuda, dims, model=None):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx]) #默认模型

    if model == None:
        model_path = '/workspace/mnt/group/personal/yangdecheng/models/nma_20200527_5T_36e.pth.tar'
        model = models.__dict__['mult5_purn_norm'](num_classes=[5,3,2,2,7])         #自己的模型
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

    if cuda:
        model.cuda()

    m1, s1, a1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)
    m2, s2, a2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)

    try:
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    except ValueError:
        # import pdb
        # pdb.set_trace()
        return 0, 2

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    fid_cosine_value = cos(a1, a2)
    return fid_value, fid_cosine_value


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #inceptionv3
    # fid_value, fid_cosine_value = calculate_fid_given_paths(args.path,
    #                                       args.batch_size,
    #                                       args.gpu != '',
    #                                       args.dims)
    # print('FID: ', fid_value)
    # print('FID_cosine: ', fid_cosine_value)

    #初始化模型---
    model_path = '/workspace/mnt/group/personal/yangdecheng/models/vehicle_property_D3C8S13_20200511.pth'
    model = models.__dict__['ProxyNetV2'](num_classes=[3,8,13])         #自己的模型
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict,strict=False)

    #inception
    # dims = 2048
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    # model = InceptionV3([block_idx]) #默认模型


    root = '/workspace/mnt/cache/MA_data_20200530/VC_data_cls8_20200530/train/van-truck'    #目标地址
    path1 = '/workspace/mnt/cache/NMA-data/fid_data/grave-shade/K095+500═т_2020-04-09-12-36-50_25477.jpg'    #基准图片
    imglit = os.listdir(root)
    for img in imglit:
        paths = []
        paths.append(path1)
        paths.append(os.path.join(root,img))
        
        fid_value, fid_cosine_value = calculate_fid_given_paths(paths,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims,
                                          model)
        print(paths[0],'\n',paths[1],fid_cosine_value)
        if fid_cosine_value <= 0.64:
            cmd = 'cp ' + str(os.path.join(root,img)) + ' /workspace/mnt/cache/NMA-data/fid_data/obscure/'
            os.system(cmd)
        elif fid_cosine_value > 0.64 and fid_cosine_value <= 0.74:
            cmd = 'cp ' + str(os.path.join(root,img)) + ' /workspace/mnt/cache/NMA-data/fid_data/mid/'
            os.system(cmd)
        elif fid_cosine_value > 0.74 and fid_cosine_value <= 1:
            cmd = 'cp ' + str(os.path.join(root,img)) + ' /workspace/mnt/cache/NMA-data/fid_data/clear/'
            os.system(cmd)
        if fid_cosine_value == 2:
            # 
            # import pdb
            # pdb.set_trace()
            continue
    print("end")
