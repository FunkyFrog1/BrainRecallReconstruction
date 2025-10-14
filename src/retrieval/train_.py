import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
##import user lib
from data_eeg import load_eeg_data
from utils import update_config, instantiate_from_config, get_device, ClipLoss, mixco_data
import time
device = get_device('auto')


def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        print(model)
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()
        self.batch_size = 1024
        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['models']['brain']['params']['z_dim']

        self.sim = np.ones((len(train_loader.dataset), 80))
        self.match_label = np.ones((len(train_loader.dataset), 80), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0
        self.match_similarities = []

        # self.load_state_dict(torch.load())

    def forward(self, batch, sample_posterior=False):
        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']
        # img = batch['img']

        # img_z = batch['img_features']
        img_z = batch['img_features'][:, :eeg.shape[1], :]
        print(img_z.shape)

        eeg_z = self.brain(eeg)

        img_z = img_z / img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = F.softplus(logit_scale)

        loss, _ = self.criterion(eeg_z.mean(dim=1), img_z.mean(dim=1), logit_scale)

        if self.config['data']['uncertainty_aware']:
            for trial_n in range(eeg_z.shape[1]):
                _, logits_per_image = self.criterion(eeg_z[:, trial_n, :], img_z[:, trial_n, :], logit_scale)
                diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
                gamma = self.gamma

                # print(self.sim.shape)
                # print(diagonal_elements.shape,  self.sim[idx][trial_n].shape)

                batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx][:, trial_n]

                mean_sim = np.mean(batch_sim)
                std_sim = np.std(batch_sim, ddof=1)
                match_label = np.ones_like(batch_sim)
                z_alpha_2 = norm.ppf(1 - self.alpha / 2)

                lower_bound = mean_sim - z_alpha_2 * std_sim
                upper_bound = mean_sim + z_alpha_2 * std_sim

                match_label[diagonal_elements > upper_bound] = 0
                match_label[diagonal_elements < lower_bound] = 2

                self.sim[idx][:, trial_n] = batch_sim
                self.match_label[idx][:, trial_n] = match_label

        return eeg_z.mean(dim=1), img_z.mean(dim=1), loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch, sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_1_accuracy = top_1_accuracy.round(3)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            top_k_accuracy = top_k_accuracy.round(3)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            # counter = Counter(self.match_label)
            counter = Counter(self.match_label.flatten().tolist())

            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            print(count_dict_mapped)
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        # label =  batch['label']
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        # compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),
                'test_top5_acc': top_k_accuracy.item(), 'mAP': self.mAP, 'similarity': self.match_similarities}

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'],
                                                                 weight_decay=1e-4)

        return [optimizer]


def main(config, yaml):
    seed_everything(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    train_loader, val_loader, test_loader = load_eeg_data(config)

    logger = TensorBoardLogger(config['save_dir'], name=config['name']+config['info'],
                               version=f"{'_'.join(config['data']['subjects'])}/seed{config['seed']}")

    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(yaml, os.path.join(logger.log_dir, yaml.rsplit('/', 1)[-1]))

    print(
        f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='max'
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='min'
        )

    trainer = Trainer(log_every_n_steps=10, #strategy=DDPStrategy(find_unused_parameters=False),
                      callbacks=[early_stop_callback, checkpoint_callback], max_epochs=config['train']['epoch'],
                      devices=[device], accelerator='cuda', logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = 'last'  # None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product


def run_experiment(args):
    eeg_backbone, vision_backbone, seed, sub, start_time, end_time = args
    yaml = "../../configs/baseline_ubp.yaml"
    config = OmegaConf.load(yaml)
    config['eeg_backbone'] = eeg_backbone
    config['vision_backbone'] = vision_backbone[0]
    config['models']['brain']['params']['z_dim'] = vision_backbone[1]
    config['data']['subjects'] = [f'sub-{(sub + 1):02d}']
    config['seed'] = seed
    config['timesteps'] = [start_time, end_time]
    config['info'] = f'-subp-[{start_time},{end_time}]-dropout0.2'
    result = main(config, yaml)


if __name__ == "__main__":
    eeg_backbones = ['Ours']
    vision_backbones = [('ViT-B-32', 512)]
    seeds = range(1)
    subs = [1]
    start_time = [250]
    end_time = [600]

    param_combinations = list(product(eeg_backbones, vision_backbones, seeds, subs, start_time, end_time))

    print(f"总共 {len(param_combinations)} 个实验")

    for params in param_combinations:
        run_experiment(params)

