# -*- coding: utf-8 -*-

# pip install librosa==0.8.0

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import librosa
import librosa.display


# Elowsson 2018

elowsson_kernel = [-106. -105., -99., -98., -95., -94., -93., -91., -90., -84., -83., -65.
, -64., -62., -61., -59., -58., -49., -48., -36., -35., -24., -23., -22.
, -21., -20., -19., -18., -17., -11., -10. , 0. , 1. , 2. , 3. , 4.
,  16.,  17.,  32.,  33.,  36.,  43.,  44.,  47.,  48.,  49.,  50.,  57.
,  65.,  66.,  67.,  68.,  72.,  74.,  75.,  76.,  78.,  80.,  81.,  83.
,  84.,  88.,  89.,  93., 101., 102., 108., 109., 110., 111., 114., 115.
, 119., 120., 121., 124., 125., 131., 132.]

# Harry Parch's system of pitches involved musical intervals that are ratios between small integers. The following kernel is of integrs up to 8, and all fractions between them.

partch_kernel = [-108. -102. -101., -94., -93., -84., -83., -72., -66., -65., -58., -57.
, -51., -50., -48., -47., -45., -44., -36., -30., -29., -27., -26., -25.
, -24., -22., -21., -18., -17., -15., -14., -12., -11., -10.,  -9.,  -8.
,  -7.,  -6.,  0.,  6.,  7.,  8.,  9.,  10.,  11.,  12.,  14.,  15.
,  17.,  18.,  21.,  22.,  24.,  25.,  26.,  27.,  29.,  30.,  36.,  44.
,  45.,  47.,  48.,  50.,  51.,  57.,  58.,  65.,  66.,  72.,  83.,  84.
,  93.,  94., 101., 102., 108.]

class SparseConv2D(nn.Module):

  def __init__(self, frames, indices):
    super(SparseConv2D, self).__init__()
    self.sparse_kernel_indices = indices 
    self.sparse_kernel_values = torch.randn(frames, indices.shape[0])
   
  def unfold_sparse_2D(self, input_tensor, indices):
    ''' indices should come in the form of a list of coordinate pairs. '''
    
    # Find the amount of zero padding needed to make the output the same
    # size as the input.
    left_pad = max(0, 0-np.min(indices[:,0]))
    top_pad = max(0, 0-np.min(indices[:,1]))
    right_pad = max(0, np.max(indices[:,0]))
    bottom_pad = max(0, np.max(indices[:,1]))
  
    input_array = input_tensor.numpy()
    padded_array = np.hstack((input_array, np.zeros((input_array.shape[0], right_pad + left_pad))))
    padded_array = np. vstack((padded_array, np.zeros((bottom_pad + top_pad, padded_array.shape[1]))))
  
    
    # Construct an array of indices for fancy indexing that slides 
    # along the input array.
    axis0_coords =  np.arange(input_array.shape[0], dtype=int) [:, np.newaxis] * np.ones(input_array.shape[1], dtype=int)
    axis1_coords =  np.ones(input_array.shape[0], dtype=int)[:, np.newaxis] * np.arange(input_array.shape[1], dtype=int)[np.newaxis, :] 
    
    axis0_ix = (axis0_coords[np.newaxis, :, :] + indices[:,1][:,np.newaxis,np.newaxis])
    axis1_ix = (axis1_coords[np.newaxis, :, :] + indices[:,0][:,np.newaxis,np.newaxis])
    return torch.tensor(padded_array[axis0_ix, axis1_ix], dtype=torch.float32)

  def forward(self, input_array):
    unfolded = torch.tensor(self.unfold_sparse_2D(input_array, self.sparse_kernel_indices), dtype = torch.float32)
    orig_shape = unfolded.shape
    linear_unfolded = unfolded.reshape(orig_shape[0], orig_shape[1] * orig_shape [2])
    linear_out = torch.mm(self.sparse_kernel_values, linear_unfolded)
    return linear_out.reshape(self.sparse_kernel_values.shape[0], orig_shape[1], orig_shape[2])

class HarmonicNet(nn.Module):
  def __init__():
    super(HarmonicNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
    self.bn2 = nn.BatchNorm2d(32)
    self.do1 = nn.Dropout(p=0.2)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
    self.bn3 = nn.BatchNorm2d(32)
    self.do2 = nn.Dropout(p=0.2)
    self.sparseConv = SparseConv2D(256, sparse_kernel)
    self.bn4 = nn.BatchNorm2d(256)
    self.mp = nn.MaxPool2d(kernel_size=(3,1), stride=(3,1), padding = (1,0))
    self.do3 = nn.Dropout(p=0.2)
    self.fc1 = nn.Linear(264 * 256, 88 * 256)
    self.bn5 = nn.BatchNorm1d(1)
    self.do4 = nn.Dropout(p=0.2)
    self.fc2 = nn.Linear(88 * 256, 88)
    self.bn6 = nn.BatchNorm1d(1)
    self.do5 = nn.Dropout(p=0.2)
    self.lstm = nn.LSTM(input_size=88, hidden_size=88, num_layers=3, dropout=0.2)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.ReLu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.ReLu(x)
    x = self.do1(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.ReLu(x)
    x = self.do2(x)
    x = self.sparseConv(x)
    x = self.bn4(x)
    x = F.ReLu(x)
    x = self.mp(x)
    x = self.do3(x)
    x = x.reshape(264 * 256)
    x = self.fc1(x)
    x = self.bn5(x)
    x = F.ReLU()
    x = self.do4(x)
    x = self.fc2(x)
    x = self.bn6(x)
    x = F.ReLu(x)
    x = self.do5(x)
    x = self.lstm(x)
    x = nn.sigmoid(x)
    return x

def get_data(train_ds, valid_ds, bs):
  return(
      DataLoader(train_ds, batch_size=bs, shuffle=True),
      DataLoader(valid_ds, batch_size=bs * 2)
  )


def get_model():
  model = HarmonicNet()
  return model, optim.SGD(model.parameters(), lr=lr, momentum = 0.9)


def loss_batch(model, loss_func, xb, yb, opt=None):
  loss = loss_func(model(xb), yb)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()
  
  return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
  for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
      loss_batch(model, loss_func, xb, yb, opt)
    
    model.eval()
    with torch.no_grad():
      losses, nums = zip(
          *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
      )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print (epoch, val_loss)

bs = 32
sparse_kernel = elowsson_kernel

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


