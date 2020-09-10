import os
import h5py as h5
import numpy as np
import math
from time import sleep
import hdmlp

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class HDMLPCam(Dataset):

    def __init__(self, source, statsfile, channels, batch_size, epochs, drop_last_batch, hdmlp_config_path,  comm_size = 1, comm_rank = 0, seed = None):
        self.source = source
        self.statsfile = statsfile
        self.batch_size = batch_size

        self.channels = channels
        self.files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') ] )
        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape

        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        transformations = [
            hdmlp.lib.transforms.Reshape(*self.data_shape),
            hdmlp.lib.transforms.ScaleShift16(data_shift, data_scale)
        ]

        self.job = hdmlp.Job(self.source,
                             batch_size,
                             epochs,
                             "uniform",
                             drop_last_batch,
                             transformations,
                             seed,
                             hdmlp_config_path,
                             filesystem_backend = "hdf5")
        self.job.setup()
        self.global_size = self.job.length()

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (data_shift.shape[0], 1, 1) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (data_scale.shape[0], 1, 1) ).astype(np.float32)

        self.comm_size = comm_size
        self.comm_rank = comm_rank

        
        #get statsfile for normalization
        #open statsfile


        if comm_rank == 0:
            print("Initialized dataset with ", self.global_size, " samples.")


    def __len__(self):
        return self.global_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def get_job(self):
        return self.job

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            raise ValueError("Must provide range in batch mode / with transformations")
        num_items = idx.stop - idx.start
        label, sample = self.job.get(num_items, True, (num_items, *self.data_shape), False, self.label_shape)
        sample = torch.from_numpy(sample)
        sample = sample.permute(0, 3, 1, 2)
        return sample, label
