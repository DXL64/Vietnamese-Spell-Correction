import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import os
from src.data.components.vocab import Vocab
from src.data.components.util import load_dataset, load_epoch_dataset
from src.data.spellcorrect_dataset import SpellCorrectDataset
from src.models.components.sampler import RandomBatchSampler, BucketBatchSampler
from src.models.components.collator import DataCollatorForCharacterTransformer as dc

from transformers import AutoTokenizer
from src.models.components.tokenizer import TokenAligner

class SpellCorrectDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        dataset: str = "dxl",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        bucket_sampling: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.bucket_sampling = bucket_sampling
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base")
        self.tokenAligner = TokenAligner(self.tokenizer)
        self.dc = dc(self.tokenAligner)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        dataset_path = os.path.join(self.hparams.data_dir, f'{self.hparams.dataset}')
        vocab_path = os.path.join(dataset_path, f'{self.hparams.dataset}.vocab.pkl')
        
        vocab = Vocab()
        vocab.load_vocab_dict(vocab_path)

        # checkpoint_dir = os.path.join(self.hparams.data_path, f'checkpoints/{args.model}')
        incorrect_file = f'{self.hparams.dataset}.train.noise'
        correct_file = f'{self.hparams.dataset}.train'
        length_file = f'{self.hparams.dataset}.length.train'

        valid_incorrect_file = f'{self.hparams.dataset}.valid.noise'
        valid_correct_file = f'{self.hparams.dataset}.valid'
        valid_length_file = f'{self.hparams.dataset}.length.valid'
        
        test_incorrect_file = f'{self.hparams.dataset}.test.noise'
        test_correct_file = f'{self.hparams.dataset}.test'
        test_length_file = f'{self.hparams.dataset}.length.test'
        
        print('Number of sequences in Train-Val-Test Dataset:')
        
        data = load_dataset(base_path=dataset_path, corr_file=correct_file, incorr_file=incorrect_file,
                        length_file = length_file)
        
        self.data_train = SpellCorrectDataset(dataset = data)
        print("Train: ", len(data))
        
        data = load_dataset(base_path=dataset_path, corr_file=valid_correct_file, incorr_file=valid_incorrect_file,
                        length_file = valid_length_file)
        
        self.data_valid = SpellCorrectDataset(dataset = data)
        print("Valid: ", len(data))
        
        data = load_dataset(base_path=dataset_path, corr_file=test_correct_file, incorr_file=test_incorrect_file,
                        length_file = test_length_file)
        
        self.data_test = SpellCorrectDataset(dataset = data)
        print("Test: ", len(data))
        
    def train_dataloader(self):
        if not self.bucket_sampling:
            train_sampler = RandomBatchSampler(self.data_train, self.hparams.batch_size)
        else:
            train_sampler = BucketBatchSampler(self.data_train)
        return DataLoader(
            dataset=self.data_train,
            batch_sampler=train_sampler,
            collate_fn=self.dc.collate,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        if not self.bucket_sampling:
            valid_sampler = RandomBatchSampler(self.data_valid, self.hparams.batch_size, shuffle = False)
        else:
            valid_sampler = BucketBatchSampler(self.data_valid, shuffle = False)
            
        return DataLoader(
            dataset=self.data_valid,
            batch_sampler=valid_sampler,
            collate_fn=self.dc.collate,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        if not self.bucket_sampling:
            valid_sampler = RandomBatchSampler(self.data_test, self.hparams.batch_size, shuffle = False)
        else:
            valid_sampler = BucketBatchSampler(self.data_test, shuffle = False)
            
        return DataLoader(
            dataset=self.data_test,
            batch_sampler=valid_sampler,
            collate_fn=dc.collate,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra, rootutils
    from omegaconf import DictConfig
    
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")
    
    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="dxl.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        datamodule: SpellCorrectDataModule = hydra.utils.instantiate(
            cfg, data_dir=f"{root}/data")
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        print('train_dataloader:', len(train_dataloader))

        batch = next(iter(train_dataloader))
        text = batch
        print("*"*10, text)

    main()
