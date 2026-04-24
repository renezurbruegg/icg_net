from __future__ import annotations
import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary


from icg_net.trainer.default import RegularCheckpointing, GraspDetection
from icg_net.utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig, load_wandb=True):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed,  workers=True)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        # print("EXPERIMENT ALREADY EXIST")
        if os.path.exists(f"{cfg.general.save_dir}/last-epoch.ckpt"):
            cfg.general["ckpt_resume_path"] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    if load_wandb:
        for log in cfg.logging:
            print(log)
            loggers.append(hydra.utils.instantiate(log))
            loggers[-1].log_hyperparams(
                flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            )
    model = GraspDetection(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        print("FOUND CHECKPOINT. Loading from ", cfg.general.checkpoint)
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


from pytorch_lightning.profilers import SimpleProfiler
 

@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg: DictConfig):
    p = SimpleProfiler()
    os.chdir(hydra.utils.get_original_cwd())
    # Save experiment config
    cfg, model, loggers = get_parameters(cfg)
    OmegaConf.save(cfg, open(os.path.join(cfg.general.save_dir, "config.yaml"), "w"), resolve = True)

    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    def restart_profiler(*args, **kargs):
        print(p.summary())
        return p

    # callbacks.append(LambdaCallback(on_train_e
    callbacks.append(RichModelSummary(max_depth=2))
    runner = Trainer(
        logger=loggers,
        callbacks=callbacks,
        # weights_save_path=str(cfg.general.save_dir),
        # profiler=p,
        # gradient_clip_val=1.0,
        # detect_anomaly=True,
        **cfg.trainer,
    )
    runner.fit(model, ckpt_path=cfg.general.ckpt_resume_path)


@hydra.main(config_path="conf", config_name="config.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        # gpus=cfg.general.gpus,
        logger=loggers,
        # weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.validate(model)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


@hydra.main(config_path="conf", config_name="config.yaml")
def get_model(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    return model, loggers
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        # weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.test(model)


if __name__ == "__main__":
    main()
