from typing import Any, List

import torch
import pyrootutils
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from lightning.pytorch.utilities.types import STEP_OUTPUT
from src.data.components.util import load_epoch_dataset
from src.data.spellcorrect_dataset import SpellCorrectDataset
from src.models.components.model import ModelWrapper
from src.models.components.tokenizer import TokenAligner
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils.metrics import get_metric_for_tfm

# from transformers.optimization import get_linear_schedule_with_warmup, AdamW
import transformers
import rootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

root = rootutils.find_root(search_from=__file__, indicator=".project-root")

class SpellCorrectionModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        optimizer: transformers.optimization,
        scheduler: transformers.optimization,
        epoch: int,
        warmup_percent: int,
        dataset_name: str,
        device: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
        
        self.data_path = str(root / "data" / dataset_name)
        self.incorrect_file = f'{dataset_name}.train.noise'
        self.correct_file = f'{dataset_name}.train'
        self.length_file = f'{dataset_name}.length.train'
        train_dataset = load_epoch_dataset(self.data_path, self.correct_file, \
            self.incorrect_file, self.length_file, 1, epoch)
        self.train_dataset = SpellCorrectDataset(dataset=train_dataset)
        
        self.total_training_steps = len(self.train_dataset) * epoch
        self.num_warmup_steps = warmup_percent * self.total_training_steps
        self.vocab_size = len(self.model_wrapper.vocab.vocab_dict['token2idx'])
        
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=self.vocab_size)
        # self.val_acc = Accuracy(task="multiclass", num_classes=self.vocab_size)
        # self.test_acc = Accuracy(task="multiclass", num_classes=self.vocab_size)
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = 0
        
        self.batch = {
            'train': [],
            'valid': [],
            'test': [],
        }

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss.reset()
        self.val_acc = []
        self.val_acc_best = 0

    def model_step(self, batch: Any):
        output = self.model(batch['batch_src'], batch['attn_masks'], batch['batch_tgt'])
        batch_loss = output['loss']
        return batch_loss, torch.from_numpy(output['preds']).to('cuda'), batch['batch_tgt'], batch['lengths']

    def training_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        loss, preds, targets, lenghts = self.model_step(batch)
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
        
        # print("----------------------------------------------")
        # print(self.decode_batch(preds), self.decode_batch(targets))

        # update and log metrics
        self.train_loss(loss)
        num_correct, num_wrong = get_metric_for_tfm(preds, targets, lenghts)
        self.train_acc.append(num_correct / (num_correct + num_wrong))
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc[-1], on_step=True, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}#, "noise": self.decode_batch(batch), "preds": self.decode_batch(preds), "targets": self.decode_batch(targets)}

    def on_train_epoch_end(self) -> None:
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        self.inference(mode='train')

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, lenghts = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        num_correct, num_wrong = get_metric_for_tfm(preds, targets, lenghts)
        self.val_acc.append(num_correct / (num_correct + num_wrong))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc[-1], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}#, "noise": self.decode_batch(batch['batch_src']), "preds": self.decode_batch(preds), "targets": self.decode_batch(targets)}

    def on_validation_epoch_end(self) -> None:
        if len(self.val_acc) > 0:   
            self.val_acc_best = max(self.val_acc) # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best, prog_bar=True)
        self.inference(mode='val')

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, lenghts = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        num_correct, num_wrong = get_metric_for_tfm(preds, targets, lenghts)
        self.test_acc.append(num_correct / (num_correct + num_wrong))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc[-1], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}#, "noise": self.decode_batch(batch), "preds": self.decode_batch(preds), "targets": self.decode_batch(targets)}

    def on_test_epoch_end(self) -> None:
        self.inference(mode='test')
        return
    
    def on_train_batch_end(self, 
                           outputs: STEP_OUTPUT, 
                           batch: Any,
                           batch_idx: int) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode='train')

    def on_validation_batch_end(self,
                                outputs: STEP_OUTPUT, 
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode ='val')

    def on_test_batch_end(self,
                          outputs: STEP_OUTPUT,
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode='test')
    
    def store_data(self, batch: Any, mode: str):
        self.batch[mode] = batch

    def inference(self, mode: str):
        data = []
        out = self.model(self.batch[mode]['batch_src'], self.batch[mode]['attn_masks'], self.batch[mode]['batch_tgt'])
        data.append([self.decode_batch(self.batch[mode]['batch_src']), self.decode_batch(out['preds']), self.decode_batch(self.batch[mode]['batch_tgt'])])

        self.logger.log_table(key=f'{mode}/infer',
                              columns=['src', 'predict', 'target'],
                              data=data)

    # def configure_optimizers(self):
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     Examples:
    #         https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    #     """
    #     optimizer = self.hparams.optimizer(params=self.parameters())
    #     if self.hparams.scheduler is not None:
    #         scheduler = self.hparams.scheduler(optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.total_training_steps)
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "num_warmup_steps": self.num_warmup_steps,
    #                 "num_training_steps": self.total_training_steps,
    #                 "monitor": "val/loss",
    #                 "interval": "epoch",
    #                 "frequency": 1,
    #             },
    #         }
    #     return {"optimizer": optimizer}
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # print(f"{self.hparams.learning_rate =}")
        optimizer = AdamW(optimizer_grouped_parameters, lr=0.0003, weight_decay=0.01, correct_bias=False)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def decode_batch(self, token_ids):
        batch = []
        for tmp in token_ids:
            batch.append(self.tokenizer.decode(tmp))
        return batch

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="model.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        module: SpellCorrectionModule = hydra.utils.instantiate(cfg)
        train_data = torch.utils.data.DataLoader(dataset=module.train_dataset,
                                    collate_fn=module.model_wrapper.collator.collate,\
                                    num_workers=2, pin_memory=True, batch_size=1)
        for step, batch in enumerate(train_data):
            out = module.model(batch['batch_src'], batch['attn_masks'], batch['batch_tgt'])
            print(out)
            break
        # inputs = ["Ô kìa, ai như cô thắm, con bác năm ở xa mới về "
        # targets
        # print(out.shape)
    
    main()
