import pytorch_lightning as pl

from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from utils.config_utils import instantiate_from_config

class DataModuleFromConfig(pl.LightningDataModule):
    """
        This function helps you wrap a torch Dataset into torch lightning Data.

    """
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, sample=False):
        """
        :param batch_size:
        :param train: train dataset config, format as following:
            train:
              target: ldm.data.brats
              params:
                data_root: xxxx
                yaml_path: xxx
                size: xx
        :param validation:
        :param test:
        :param predict:
        :param wrap: If the incoming data is not of dataset class but has len and getitem attributes, it can also be wrapped.
        :param num_workers:
        :param shuffle_test_loader:
        :param use_worker_init_fn:
        :param shuffle_val_dataloader:
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = 60
        # self.num_workers = 0
        self.use_worker_init_fn = use_worker_init_fn
        self.sample = sample
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.datasets = None

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        print('***********************************************************')
        print('batch size is setting: %d'%self.batch_size)
        if self.sample:
            assert hasattr(self.datasets["train"].data, 'weight_list')
            weights = torch.DoubleTensor(self.datasets["train"].data.weight_list)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, sampler=sampler)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]