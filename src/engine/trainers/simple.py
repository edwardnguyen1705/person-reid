import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import torch
import numpy as np
from multiprocessing import Manager
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.dataloader import DataLoader

from engine import hooks
from engine.trainers.base import BaseTrainer
from data import (
    DatasetCache,
    build_datasource,
    build_transform,
    ReidDataset,
    DataPrefetcher,
    RandomIdentitySampler,
    DataLoaderX,
)
from models import build_model
from optimizers import build_optimizers
from losses import WrapLoss
from utils import ModelEmaV2, check_any
from metrics import (
    accuracy,
    cosine_dist,
    euclidean_dist,
    hamming_distance,
    evaluate_rank,
)
from extractor import feature_extractor


class SimpleTrainer(BaseTrainer):
    def __init__(self, args, cfg, device):
        super(SimpleTrainer, self).__init__(args, device)
        self.cfg = cfg

        # Step 2: Create datasource, dataloader
        self.datasource = self.build_datasource()
        self.train_dataloader = self.build_train_loader()
        self.query_dataloader, self.gallery_dataloader = self.build_test_loader()

        # Step 3: Scale numclasses
        self.auto_scale_hyperparams()

        # Step 4: Create model
        self.model = self.build_model().to(self.device)

        # Step 5: Create optimizer
        self.optimizer = self.build_optimizers()

        # Step 5: Create criterion
        self.criterion = self.build_losses().to(self.device)

        # Step 6: Create grad scaler
        self.scaler = GradScaler(
            enabled=torch.cuda.is_available() and self.cfg["trainer"]["amp"]
        )

        self.lr_scheduler = hooks.LrScheduler(
            self.cfg["lr_scheduler"],
            self.optimizer,
            total_iterations=len(self.train_dataloader),
        )

        # Step ?: Create ema model
        self.model_ema = None
        if self.cfg["ema"]["enable"]:
            self.model_ema = ModelEmaV2(
                self.model, decay=self.cfg["ema"]["decay"], device=None
            )

        self.checkpointer = hooks.Checkpointer(
            train_metrics=self.get_train_metric_dict(),
            val_metrics=self.get_val_metric_dict(),
            checkpoint_dir=self.args.checkpoint_dir,
        )

        self.register_hooks(self.build_hooks())

        # Step ?:
        self.start_epoch = 1
        if self.args.resume_path != "":
            self.checkpointer.resume_from_checkpoint(self.args.resume_path, self.device)

    def build_datasource(self):
        return build_datasource(
            cfg=self.cfg["data"],
            data_root=self.args.data_root,
        )

    def build_train_loader(self):
        transform = build_transform(
            self.cfg["data"]["image_size"],
            is_training=True,
            use_autoaugmentation=self.cfg["data"]["train"]["transform"][
                "autoaugmentation"
            ],
            use_random_erasing=self.cfg["data"]["train"]["transform"]["random_erasing"],
            use_cutout=self.cfg["data"]["train"]["transform"]["cutout"],
            use_random2translate=self.cfg["data"]["train"]["transform"][
                "random2translate"
            ],
            use_lgt=self.cfg["data"]["train"]["transform"]["lgt"],
            use_random_grayscale=self.cfg["data"]["train"]["transform"][
                "random_grayscale"
            ],
        )
        cache = None
        if self.args.on_memory_dataset:
            manager = Manager()
            cache = DatasetCache(manager)
        dataset = ReidDataset(
            data=self.datasource.get_data("train"),
            cache=cache,
            transform=transform,
            transform_lib="torchvision",
        )

        sampler = RandomIdentitySampler(
            self.datasource.get_data("train"),
            batch_size=self.cfg["data"]["train"]["batch_size"],
            num_instances=self.cfg["data"]["train"]["num_instances"],
        )

        dataloader_args = {
            "dataset": dataset,
            "sampler": sampler,
            "batch_size": self.cfg["data"]["train"]["batch_size"],
            "num_workers": self.cfg["data"]["num_workers"],
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
        }

        if self.args.background_generator:
            return DataLoaderX(device=self.device, **dataloader_args)
        return DataLoader(**dataloader_args)

    def build_test_loader(self):
        transform = build_transform(self.cfg["data"]["image_size"], is_training=False)

        query_dataset = ReidDataset(
            data=self.datasource.get_data("query"),
            transform=transform,
            transform_lib="torchvision",
        )

        gallery_dataset = ReidDataset(
            data=self.datasource.get_data("gallery"),
            transform=transform,
            transform_lib="torchvision",
        )

        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.cfg["data"]["test"]["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        gallery_dataloader = DataLoader(
            gallery_dataset,
            batch_size=self.cfg["data"]["test"]["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        return query_dataloader, gallery_dataloader

    def build_model(self):
        return build_model(self.cfg["model"], len(self.datasource.get_classes()))

    def build_optimizers(self):
        return build_optimizers(self.cfg["optimizer"], self.model)

    def build_losses(self):
        return WrapLoss(self.cfg["loss"])

    def build_hooks(self):
        ret = [
            self.lr_scheduler,
        ]

        if self.cfg["freeze"]["enable"]:
            ret.append(
                hooks.FreezeLayers(
                    self.model,
                    self.cfg["freeze"]["layers"],
                    self.cfg["freeze"]["epochs"],
                )
            )

        ret.extend(
            [
                hooks.Writer(total_iterations=len(self.train_dataloader)),
                hooks.Wandb(
                    config={**self.cfg, **vars(self.args)},
                    project=self.cfg["wandb"]["project"],
                    run_id=self.args.run_id,
                    entity=self.cfg["wandb"]["entity"],
                    group=None,
                    sync_tensorboard=True,
                ),
                self.checkpointer,
            ]
        )

        return ret

    # def auto_scale_hyperparams(self):
    # self.cfg["model"]["num_classes"] = len(self.datasource.get_classes())

    def get_train_metric_dict(self):
        return {
            "loss": True,
            "id_loss": True,
            "ranking_loss": True,
            "top1": False,
            "top5": False,
            "top10": False,
        }

    def get_val_metric_dict(self):
        return {
            "mAP": False,
            "mINP": False,
            "top1": False,
            "top5": False,
            "top10": False,
        }

    def train(self):
        super().train(self.start_epoch, self.cfg["trainer"]["epochs"])

    def before_train_epoch(self):
        super().before_train_epoch()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.prefetcher = DataPrefetcher(
            self.train_dataloader,
            self.device,
            enable=self.args.data_prefetcher and torch.cuda.is_available(),
        )

    def run_train_step(self):
        data, target, *_ = next(self.prefetcher)

        with autocast(enabled=torch.cuda.is_available() and self.cfg["trainer"]["amp"]):
            feat, score, pred_cls = self.model(data, target)

            check_any(feat)
            check_any(score)
            check_any(pred_cls)

            loss, loss_item = self.criterion(feat, score, target)

            check_any(loss)

        self.scaler.scale(loss).backward()

        if (self.batch_idx + 1) % self.cfg["iters_to_accumulate"] == 0:
            # Clips gradient norm of an iterable of parameters.
            if self.cfg["clip_grad_norm_"]["enable"]:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfg["clip_grad_norm_"]["max_norm"],
                )

            # optimize
            self.scaler.step(self.optimizer)

            if self.model_ema is not None:
                self.model_ema.update(self.model)

            # Updates the scale for next iteration.
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

        if isinstance(pred_cls, (list, tuple)):
            total_acc1, total_acc5, total_acc10 = 0, 0, 0
            for x in pred_cls:
                acc1, acc5, acc10 = accuracy(x, target, topk=(1, 5, 10))
                total_acc1, total_acc5, total_acc10 = (
                    total_acc1 + acc1,
                    total_acc5 + acc5,
                    total_acc10 + acc10,
                )
            acc1, acc5, acc10 = (
                total_acc1 / len(pred_cls),
                total_acc5 / len(pred_cls),
                total_acc10 / len(pred_cls),
            )
        elif isinstance(pred_cls, torch.Tensor):
            acc1, acc5, acc10 = accuracy(pred_cls, target, topk=(1, 5, 10))
        else:
            raise RuntimeError("pred_cls has type not support")

        return {
            "loss": loss.item(),
            "id_loss": loss_item[0].item(),
            "ranking_loss": loss_item[1].item(),
            "top1": acc1.item(),
            "top5": acc5.item(),
            "top10": acc10.item(),
        }

    def check_is_val_epoch(self, epoch: int):
        if self.args.val and epoch >= self.args.val_step:
            if epoch % self.args.val_step == 0:
                return True

            if epoch >= self.cfg["trainer"]["epochs"] * 0.9:
                return True

        return False

    def before_val_epoch(self):
        self.model_test = (
            self.model_ema.module
            if ((self.model_ema != None) and self.cfg["testing"]["test_on_ema"])
            else self.model
        )

    def run_val_epoch(self):
        self.model_test.eval()

        query_feature, query_label, query_camera = feature_extractor(
            self.model_test,
            self.query_dataloader,
            self.device,
            description="Query",
            flip_inference=self.cfg["testing"]["flip_inference"],
        )

        gallery_feature, gallery_label, gallery_camera = feature_extractor(
            self.model_test,
            self.gallery_dataloader,
            self.device,
            description="Gallery",
            flip_inference=self.cfg["testing"]["flip_inference"],
        )

        if self.cfg["testing"]["distance_mode"] == "euclidean":
            distance = euclidean_dist(
                query_feature, gallery_feature, sqrt=False, clip=False
            )
        elif self.cfg["testing"]["distance_mode"] == "cosine":
            distance = cosine_dist(query_feature, gallery_feature, alpha=1)
        elif self.cfg["testing"]["distance_mode"] == "hamming":
            distance = hamming_distance(query_feature, gallery_feature)
        else:
            raise ValueError(
                "cfg[testing][distance_mode] must be one of euclidean, cosine, hamming"
            )

        cmc, all_AP, all_INP = evaluate_rank(
            distance, query_label, gallery_label, query_camera, gallery_camera
        )

        return {
            "mAP": np.mean(all_AP),
            "mINP": np.mean(all_INP),
            "top1": cmc[0] * 100,
            "top5": cmc[4] * 100,
            "top10": cmc[9] * 100,
        }

    def check_is_test(self):
        return bool(self.args.test_from_checkpoint)

    def before_test_step(self):
        super().before_test_step()
        self.model_test = (
            self.model_ema.module
            if ((self.model_ema != None) and self.cfg["testing"]["test_on_ema"])
            else self.model
        )

    def run_test(self):
        if self.args.test_from_checkpoint:
            print("Testing on each checkpoint")
            for _ in self.checkpointer.resume_to_test():
                self.before_test_step()
                self.result_test_step = self.run_val_epoch()
                super().after_test_step()
