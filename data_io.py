"""
Functions for dealing with data input and output.
"""

import os
import numpy as np
import pickle
import json
import struct

import sys

from tqdm import tqdm

from fgnt.mask_estimation import estimate_IBM
from fgnt.signal_processing import audioread
from fgnt.signal_processing import stft

#-----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def shrink_to_min(feats, out_shape):

    default = feats['clean'] if 'clean' in feats else feats['noisy']

    min_len = np.inf
    for i in range(len(default)):
        length = default[i].shape[-2]
        if length < min_len:
            min_len = length

    indexes = []
    for i in range(len(default)):
        length = default[i].shape[-2]
        feats['frames'] = length
        if length > min_len:
            idx = np.random.randint(length - min_len)

            for name in ['clean', 'noisy'] & feats.keys():
                feats[name][i] = feats[name][i][:,idx:idx+min_len]
        else:
            idx = 0
            pad_shape = ((0,0),(0,min_len-length),(0,0))

            for name in ['clean', 'noisy'] & feats.keys():
                feats[name][i] = np.pad(feats[name][i], pad_shape, mode='edge')

        if 'senone' in feats:
            senone_len = len(feats['senone'][i])
            feats['senone_length'] = senone_len
            if senone_len > min_len+idx:
                feats['senone'][i] = feats['senone'][i][idx:idx+min_len]
            elif senone_len > min_len:
                feats['senone'][i] = feats['senone'][i][:min_len]
            else:
                feats['senone'][i] = np.pad(feats['senone'][i], ((0,min_len-senone_len)), mode='edge')

        indexes.append(idx)

    for name in ['clean', 'noisy', 'senone'] & feats.keys():
        feats[name] = np.array(feats[name], dtype = np.float32)

        if name == 'clean' or name == 'noisy':
            feats[name] = feats[name].reshape((out_shape[0], out_shape[1], min_len, -1))
        elif out_shape[0] == 1:
            feats[name] = feats[name].reshape((out_shape[0], out_shape[1], min_len))
        else:
            feats[name] = np.tile(feats[name], (out_shape[0], out_shape[1], 1))

    return feats

def load_arrays_from_numpy(base_dir, flist, idx_list):

    min_len = np.inf
    feats = {'clean':[], 'noisy':[]}
    for i, f in enumerate(flist):
        data = np.load(os.path.join(base_dir, f))
        feats['clean'].append(np.squeeze(data['clean'],axis=0)[idx_list[i]])
        feats['noisy'].append(np.squeeze(data['noisy'],axis=0)[idx_list[i]])

        if 'senone' in data:
            if 'senone' not in feats:
                feats['senone'] = []
                feats['senone_length'] = []

            #if len(idx_list[i]) > 1:
            #    senone = np.tile(data['senone'], (len(idx_list[i]), 1))
            #else:
            senone = np.squeeze(data['senone'])

            feats['senone'].append(senone)
            feats['senone_length'].append(data['senone_length'])

        length = data['clean'].shape[-2]
        if length < min_len:
            min_len = length

    return feats

def load_arrays_from_wav(base_dir, flist, idx_list, delay = 0):

    kwargs = {'time_dim': 1, 'size':512, 'shift':160, 'window_length':400}

    feats = []
    for i, f in enumerate(flist):
        utt = []
        for idx in idx_list[i]:

            # Clean data has one channel, but need to replicate it
            if idx >= len(f):
                idx = 0

            filename = os.path.join(base_dir, f[idx])
            audio = np.expand_dims(audioread(filename), axis = 0)
            if delay > 0:
                audio = np.roll(audio, delay, axis = -1)

            if audio.ndim == 3:
                feat = np.abs(stft(audio[:,0], **kwargs)) / 2
                feat += np.abs(stft(audio[:,1],**kwargs)) / 2
            else:
                feat = np.abs(stft(audio, **kwargs))

            utt.append(feat)

        feats.append(np.concatenate(utt))

    return feats

def load_arrays_from_scp(base_dir, flist, remove_deltas = False, divisor = 16):
    feats = []
    for utterance in flist:
        utt = []
        for fname, offset in utterance:
            with open(os.path.join(base_dir, fname), 'rb') as f:
                f.seek(offset)
                header = struct.unpack("<xcccc", f.read(5))
                _,rows = struct.unpack("<bi", f.read(5))
                _,cols = struct.unpack("<bi", f.read(5))
                mat = np.frombuffer(f.read(rows*cols*4), dtype=np.float32)
                mat = np.reshape(mat, (rows, cols))
                if remove_deltas:
                    mat = mat[:,:cols//3]

                if divisor > 1:
                    mat = mat[:,:-(mat.shape[1] % divisor)]
                    pad = ((0,divisor - rows % divisor),(0,0))
                    mat = np.pad(mat, pad, 'edge')

            utt.append(mat)
        feats.append(np.stack(utt))
    return feats

def kaldi_write_mats(ark_path, utt_id, utt_mat):
    ark_write_buf = open(ark_path, "ab")
    utt_mat = np.asarray(utt_mat, dtype=np.float32)
    rows, cols = utt_mat.shape
    ark_write_buf.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
    ark_write_buf.write(struct.pack('<cxcccc', b' ',b'B',b'F',b'M',b' '))
    ark_write_buf.write(struct.pack('<bi', 4, rows))
    ark_write_buf.write(struct.pack('<bi', 4, cols))
    ark_write_buf.write(utt_mat)

def read_txt(f):

    dictionary = {}
    for line in f:
        line = line.split()
        dictionary[line[0]] = np.array(line[1:], dtype=np.int32)

    return dictionary

def read_scp(f):
    dictionary = {}
    for line in f:
        if line.strip() == '':
            continue

        uttid, filename, byte = line.split()
        if uttid not in dictionary:
            dictionary[uttid] = [(filename, int(byte))]
        else:
            dictionary[uttid].append((filename, int(byte)))

    return dictionary

class DataLoader:
    """ Class for loading features and senone labels from file into a buffer, and batching. """

    def __init__(self,
        base_dir,
        flists = [('numpy', 'json', 'numpy.flist')],
        compute_ibm = False,
        compute_irm = False,
        shuffle     = False,
        logify      = False,
        channels    = 1,
        batch_size  = 1,
        stage       = "tr", # or "dt" or "ev"
    ):
        """ Initialize the data loader including filling the buffer """

        self.base_dir = base_dir
        self.compute_ibm = compute_ibm
        self.compute_irm = compute_irm
        self.shuffle = shuffle
        self.logify  = logify
        self.channels = channels
        self.batch_size = batch_size

        self.flists = {}
        for name, ftype, flist in flists:
            with open(os.path.join(base_dir, stage, flist)) as f:

                if ftype == 'json':
                    data = json.load(f)
                elif ftype == 'txt':
                    data = read_txt(f)
                elif ftype == 'scp':
                    data = read_scp(f)
                else:
                    raise ValueError("Type must be one of 'json', 'txt', 'scp'")

                self.flists[name] = {'type': ftype, 'data': data}


        # Get the ids and 
        if 'numpy' in self.flists:
            self.ids = list(self.flists['numpy']['data'].keys())
            arrays = np.load(os.path.join(base_dir, self.flists['numpy']['data'][self.ids[0]]))
            self.available_channels = arrays['clean'].shape[1]
        else:
            name = 'noisy' if 'noisy' in self.flists else 'clean'
            self.ids = list(self.flists[name]['data'].keys())
            self.available_channels = len(self.flists[name]['data'][self.ids[0]])

    def batchify(self):
        """ Iterate through batches """
        
        n = len(self.ids)
        indexes = np.random.permutation(n) if self.shuffle else np.arange(n)
        if len(indexes) % self.batch_size != 0:
            indexes = indexes[:-(len(indexes) % self.batch_size)]
        indexes = indexes.reshape((-1, self.batch_size))

        for batch_idxs in tqdm(indexes):
            ids = [self.ids[i] for i in batch_idxs]
            batch = self.get_batch(ids, divisor = 16)
            batch['ids'] = ids

            #batch, feat_length = self.get_batch(batch['ids'])
            #batch, new_length = self.normalize_batch(batch, feat_length, divisor = 16)
            
            #batch['added'] = new_length - feat_length
            #batch['frames'] = new_length

            yield batch

    def get_batch(self, uttids, divisor = 16):
        """ Load a batch of data from files """

        batch = {}

        # Pick channels
        if self.shuffle:
            channel_idx = np.random.randint(self.available_channels, size = (self.batch_size, self.channels))
            out_shape = (self.batch_size, self.channels)
        else:
            channel_idx = np.tile(np.arange(self.available_channels), (self.batch_size, 1))
            out_shape = (self.batch_size * (self.available_channels // self.channels), self.channels)


        if 'numpy' in self.flists:
            flist = [self.flists['numpy']['data'][uid] for uid in uttids]
            feats = load_arrays_from_numpy(self.base_dir, flist, channel_idx)
        else:
            feats = {}
            start_idx = None
            for name in ['clean', 'noisy', 'noise'] & self.flists.keys():
                
                flist = [self.flists[name]['data'][uid] for uid in uttids]
                if self.flists[name]['type'] == 'json':
                    feats[name], start_idx = load_arrays_from_wav(self.base_dir, flist, channel_idx)
                elif self.flists[name]['type'] == 'scp':
                    feats[name] = load_arrays_from_scp(self.base_dir, flist, remove_deltas = name == 'noisy')
                else:
                    raise ValueError("Type must be one of 'json', 'scp'")

        if 'senone' in self.flists:
            feats['senone'] = [self.flists['senone']['data'][u] for u in uttids]

        feats = shrink_to_min(feats, out_shape)

        if self.compute_ibm:
            if 'noise' in feats and 'clean' in feats:
                feats['ibm_x'], feats['ibm_n'] = estimate_IBM(feats['clean'], feats['noise'])
            elif 'noisy' in feats and 'clean' in feats:
                feats['ibm_x'], feats['ibm_n'] = estimate_IBM(feats['clean'], feats['noisy'], -15, -15)
            else:
                raise ValueError("To compute IBM, clean and noise or noisy signals are required")

        if self.compute_irm:
            if 'noise' in feats and 'clean' in feats:
                feats['irm'] = feats['clean'] / (feats['clean'] + feats['noise'])
                #feats['irm'][feats['irm'] > 1] = 1
            elif 'noisy' in feats and 'clean' in feats:
                if np.min(feats['clean']) < 0:
                    minimum = min(np.min(feats['clean']), np.min(feats['noisy']))
                    feats['irm'] = (feats['clean'] - minimum + 1e-6) / (feats['noisy'] - minimum + 1e-6)
                    feats['irm'] /= 1.3
                else:
                    feats['irm'] = np.sqrt(feats['clean']) / np.sqrt(feats['noisy'])
                    feats['irm'] /= 2
                feats['irm'][feats['irm'] > 1] = 1
            else:
                raise ValueError("To compute IRM, clean and noise or noisy signals are required")

        if 'noise' in feats:
            feats['noisy'] = feats['noise'] + feats['clean']

        # Reshape features
        for name in ['clean', 'noisy'] & feats.keys():
            if self.logify:
                feats[name] = np.log(feats[name] + 0.01)

        if 'trans' in self.flists:
            indices = np.array([(0,i) for i in range(len(self.flists['trans'][uttid]))], dtype=np.int32)
            values = np.array(self.flists['trans'][uttid], dtype=np.int32)
            shape = np.array((1, len(self.flists['trans'][uttid])), dtype=np.int32)
            feats['trans'] = (indices, values, shape)

        return feats

