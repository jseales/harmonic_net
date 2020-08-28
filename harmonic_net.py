# -*- coding: utf-8 -*-
class SparseConv1D(nn.Module):
  
  def __init__(self, sk_ind, in_channels=1, out_channels=1, device='cpu'):
    super(SparseConv1D, self).__init__()
    self.out_channels = out_channels
    self.in_channels = in_channels 
    self.sk_ind = np.array(sk_ind, dtype=int)
    self.sk_len = len(sk_ind)
    self.sk_weights = torch.randn(out_channels, in_channels, self.sk_len, 
                                  dtype=torch.float, requires_grad=True, device=device)
    self.device = device
    #print('self.sk_weights\n', self.sk_weights)


  def unfold_sparse_1D(self, input_tensor):
    # Find the amount of zero padding needed to make the output the same
    # size as the input.
    # print('input_tensor.shape', input_tensor.shape)
    low_pad = int(max(0 - min(self.sk_ind), 0))
    high_pad = int(max(0, max(self.sk_ind)))
    input_array = input_tensor.cpu().detach().numpy()
    padded_array = np.hstack((input_array, 
                              np.zeros((self.in_channels, high_pad)), 
                              np.zeros((self.in_channels, low_pad))))
    # print('padded array\n', padded_array)

    # Construct an array of indices that will be used to make the 
    # unfolded array via numpy fancy indexing. 
    # Broadcast to make an array of shape(sk_len, input_len)
    indices = self.sk_ind[:, np.newaxis] + np.arange(self.input_len)
    # print('indices\n', indices)
    # output of array has shape(in_channels, sk_len, input_len)
    return torch.tensor(padded_array[np.arange(self.in_channels)[:, np.newaxis, np.newaxis], 
                                     indices[np.newaxis, :, :]], 
                                     dtype=torch.float, device=self.device)

  def forward(self, input_tensor):
    batch_size = input_tensor.shape[0]
    self.input_len = input_tensor.shape[2]
    output_batch = torch.empty(batch_size, self.out_channels, self.input_len,
                               dtype=torch.float, device=self.device)
    
    for i in range(batch_size):
    # Input_array will come in shape (in_channels, input_len)
      unfolded = self.unfold_sparse_1D(input_tensor[i])
      # print('unfolded\n', unfolded)
      #print(self.sk_weights)
      output_batch[i] = torch.mm(self.sk_weights.reshape(self.out_channels, self.in_channels * self.sk_len), 
                      unfolded.reshape(self.in_channels * self.sk_len, self.input_len))
    return output_batch

class HarmonicNet(nn.Module):
  def __init__(self):
    CD = 32 # "convolution depth" originally set to 32
    SCD = 256 # "sparse convolution depth" originally set to 256
    super(HarmonicNet, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=CD, kernel_size=3, stride=1, padding=1, padding_mode='zeros').to(device)
    self.bn1 = nn.BatchNorm1d(CD).to(device)
    self.conv2 = nn.Conv1d(in_channels=CD, out_channels=CD, kernel_size=3, stride=1, padding=1, padding_mode='zeros').to(device)
    self.bn2 = nn.BatchNorm1d(CD).to(device)
    self.do1 = nn.Dropout(p=0.2).to(device)
    self.conv3 = nn.Conv1d(in_channels=CD, out_channels=CD, kernel_size=3, stride=1, padding=1, padding_mode='zeros').to(device)
    self.bn3 = nn.BatchNorm1d(CD).to(device)
    self.do2 = nn.Dropout(p=0.2).to(device)
    self.sparseConv = SparseConv1D(sparse_kernel, in_channels=CD, out_channels=SCD, device='cuda:0').to(device)
    self.bn4 = nn.BatchNorm1d(SCD).to(device)
    self.mp = nn.MaxPool1d(kernel_size=3, stride=3, padding=0).to(device)
    self.do3 = nn.Dropout(p=0.2).to(device)
    self.fc1 = nn.Linear(112 * SCD, 88 * SCD).to(device)
    self.bn5 = nn.BatchNorm1d(88 * SCD).to(device)
    self.do4 = nn.Dropout(p=0.2).to(device)
    self.fc2 = nn.Linear(88 * SCD, 88).to(device)
    self.bn6 = nn.BatchNorm1d(88).to(device)
    self.fc3 = nn.Linear(88, 12).to(device)
    self.bn7 = nn.BatchNorm1d(12).to(device)
    self.do5 = nn.Dropout(p=0.2).to(device)
    #self.lstm = nn.LSTM(input_size=12, hidden_size=12, num_layers=3, dropout=0.2).to(device)


  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.do1(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.do2(x)
    x = self.sparseConv(x)
    x = self.bn4(x)
    x = F.relu(x)
    x = self.mp(x)
    x = self.do3(x)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x = self.fc1(x)
    x = self.bn5(x)
    x = F.relu(x)
    x = self.do4(x)
    x = self.fc2(x)
    x = self.bn6(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = self.bn7(x)
    x = F.relu(x)
    x = self.do5(x)
    #x = self.lstm(x)
    x = torch.sigmoid(x)
    return x


class MiniHarmonicNet(nn.Module):
  def __init__(self):
    CD = 4 # "convolution depth" originally set to 32
    SCD = 16 # "sparse convolution depth" originally set to 256
    super(MiniHarmonicNet, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=CD, kernel_size=3, stride=1, padding=1, padding_mode='zeros').to(device)
    self.bn1 = nn.BatchNorm1d(CD).to(device)
    self.do1 = nn.Dropout(p=0.2).to(device)
    self.sparseConv = SparseConv1D(sparse_kernel, in_channels=CD, out_channels=SCD, device='cuda:0').to(device)
    self.bn4 = nn.BatchNorm1d(SCD).to(device)
    self.mp = nn.MaxPool1d(kernel_size=3, stride=3, padding=0).to(device)
    self.do3 = nn.Dropout(p=0.2).to(device)
    self.fc1 = nn.Linear(112 * SCD, 88 * SCD).to(device)
    self.bn5 = nn.BatchNorm1d(88 * SCD).to(device)
    self.do4 = nn.Dropout(p=0.2).to(device)
    self.fc2 = nn.Linear(88 * SCD, 88).to(device)
    self.bn6 = nn.BatchNorm1d(88).to(device)
    self.fc3 = nn.Linear(88, 12).to(device)
    self.bn7 = nn.BatchNorm1d(12).to(device)
    self.do5 = nn.Dropout(p=0.2).to(device)
    #self.lstm = nn.LSTM(input_size=12, hidden_size=12, num_layers=3, dropout=0.2).to(device)


  def forward(self, x):
    # print('x = x.unsqueeze(1)')
    x = x.unsqueeze(1)
    # print('x = self.conv1(x)')
    x = self.conv1(x)
    # print('x = self.bn1(x)')
    x = self.bn1(x)
    # print('x = F.relu(x)')
    x = F.relu(x)
    # print('x = self.do1(x)')
    x = self.do1(x)
    # print('x = self.sparseConv(x)')
    x = self.sparseConv(x)
    # print('x = self.bn4(x)')
    x = self.bn4(x)
    # print('x = F.relu(x)')
    x = F.relu(x)
    # print('x = self.mp(x)')
    x = self.mp(x)
    # print('x = self.do3(x)')
    x = self.do3(x)
    # print('x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])')
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    # print('x = self.fc1(x)')
    x = self.fc1(x)
    # print('x = self.bn5(x)')
    x = self.bn5(x)
    # print('x = F.relu(x)')
    x = F.relu(x)
    # print('x = self.do4(x)')
    x = self.do4(x)
    # print('x = self.fc2(x)')
    x = self.fc2(x)
    # print('x = self.bn6(x)')
    x = self.bn6(x)
    # print('x = F.relu(x)')
    x = F.relu(x)
    # print('x = self.fc3(x)')
    x = self.fc3(x)
    # print('x = self.bn7(x)')
    x = self.bn7(x)
    # print('x = F.relu(x)')
    x = F.relu(x)
    # print('x = self.do5(x)')
    x = self.do5(x)
    #print('x = self.lstm(x)')
    #x = self.lstm(x)
    # print('x = torch.sigmoid(x)')
    x = torch.sigmoid(x)
    return x


(sig, rate) = librosa.load('./MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.wav', sr=None)
midi_data = pm.PrettyMIDI('./MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi')

def frequency_step(time_resolution, bins_per_octave, lowest_note):
  lowest_frequency = 440 * (2 ** ((lowest_note-69) / 12))
  ratio_between_bins = 2 ** (1/bins_per_octave) - 2 ** (-1/bins_per_octave)
  biggest_frequency_step = lowest_frequency * ratio_between_bins
  maximum_bandwith = 2.88 / time_resolution
  gamma = maximum_bandwith - biggest_frequency_step
  return gamma

gamma = frequency_step(0.5, 36, 24)
gamma

# 1/3 of a semitone below note 21.
fmin = librosa.core.midi_to_hz(20.66667)
fmin

spec_mag_db = librosa.core.power_to_db(
    np.abs(
        librosa.vqt(sig, 
                    sr=rate, 
                    fmin=fmin, 
                    n_bins=336, 
                    gamma=gamma, 
                    bins_per_octave=36).T
    )
)

print(spec_mag_db.shape)

X = spec_mag_db - np.mean(spec_mag_db)
X = X / np.std(X)

print(X.shape)

y = midi_data.get_chroma(fs=93.75).T

if X.shape[0] > y.shape[0]:
  X = X[:y.shape[0], :]
elif X.shape[0] < y.shape[0]:
  y = y[:X.shape[0], :]

#X = X[:, np.newaxis, :]

print (X.shape, y.shape)

device = torch.device('cuda:0')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=False) 
X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=False)
y_train = torch.tensor(y_train, dtype=torch.float, requires_grad=False) 
y_test = torch.tensor(y_test, dtype=torch.float, requires_grad=False)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device) 
y_test = y_test.to(device)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def get_data(train_ds, valid_ds, bs):
  return(
      DataLoader(train_ds, batch_size=bs, shuffle=True),
      DataLoader(valid_ds, batch_size=bs * 2)
  )


def get_model(lr=0.001):
  model = HarmonicNet()
  return model, optim.SGD(model.parameters(), lr=lr, momentum = 0.9)

def loss_batch(model, loss_func, xb, yb, opt=None):
  mxb = model(xb)
  # print('mxb.shape', mxb.shape)
  # print('yb.shape', yb.shape)
  loss = loss_func(mxb, yb)

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
# loss_func = torch.nn.MSELoss(reduction='sum')
loss_func = nn.BCEWithLogitsLoss()
epochs = 10

train_dl, test_dl = get_data(train_ds, test_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, test_dl)

