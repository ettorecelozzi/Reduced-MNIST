import numpy as np
import torch
from torchvision.datasets import MNIST
from collections import defaultdict
import pickle as pkl
import h5py


def get_rmnist_idx(n, labels, test=False):
    """
    Generate indexes to get n elemnt for each digits
    :param n: number of element (int)
    :param labels: list of train labels
    :return: dictionary where the key is the label and the values are the indexes of the samples
    """
    indices = defaultdict(list)
    for i, label in enumerate(labels):
        if len(indices[label]) < n:
            indices[label].append(i)
    filename = "./RMNIST/indexes/rmnist" + str(n) + "_train" if test is False else "./RMNIST/indexes/rmnist" + str(
        n) + "_test"
    with open(filename, "wb") as f:
        pkl.dump(indices, f)
    return indices


def get_rmnist(n, indices, imgs, labels, test=False):
    """
    Retrieve the samples given the dictionary of the samples indexes
    :param n: number of samples for each digit
    :param indices: dictionary (see get_rmnist_idx)
    :param imgs: full dataset of images
    :param labels: full dataset of labels
    """
    imgs_r = []
    labels_r = []
    for key in indices.keys():
        for idx in indices[key]:
            imgs_r.append(imgs[idx])
            labels_r.append(labels[idx])
    filename = './RMNIST/dataset/trainImg_' + str(
        n) + '.npy' if test is False else './RMNIST/dataset/testImg_' + str(n) + '.npy'
    filename_label = './RMNIST/dataset/trainlabel_' + str(
        n) + '.npy' if test is False else './RMNIST/dataset/testlabel_' + str(n) + '.npy'
    np.save(filename, np.asarray(imgs_r))
    np.save(filename_label, np.asarray(labels_r))


# Load MNIST train and test sets.
ds_trainX = MNIST(root='./', train=True, download=True)
ds_trainY = ds_trainX.targets.numpy()
ds_trainX = ds_trainX.data.numpy().astype('float32')
ds_testX = MNIST(root='./', train=False, download=True)
ds_testY = ds_testX.targets.numpy()
ds_testX = ds_testX.data.numpy().astype('float32')
print(ds_trainX[0].shape)

n = 40
indices = get_rmnist_idx(n, ds_trainY, test=False)
with open("./RMNIST/indexes/rmnist" + str(n) + "_train", "rb") as f:
    data = pkl.load(f)
get_rmnist(n, data, ds_trainX, ds_trainY, test=False)

n = 25
indices_test = get_rmnist_idx(n, ds_testY, test=True)
with open("./RMNIST/indexes/rmnist" + str(n) + "_test", "rb") as f:
    data_test = pkl.load(f)
get_rmnist(n, data_test, ds_testX, ds_testY, test=True)
