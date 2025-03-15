#!/usr/bin/env python
# coding=utf-8

import copy
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

# from models.ema_model import EMAModel
# from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
# from models.multimodal_encoder.t5_encoder import T5Embedder
from models.DMP.dmp_runner import DMPRunner
# from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
# from train.sample import log_sample_res

if is_wandb_available():
    import wandb

def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # Set up logging
    logging_dir = Path(args.output_dir, args.logging_dir)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Set up accelerator
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Set seed before initializing model.
    if args.seed is not None:
        set_seed(args.seed)

    # create output directory if needed
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # set the weights dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Vision encoder
    # vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    # vision_encoder = 
    # image_processor = vision_encoder.image_processor
    
    # Load model from checkpoint if provided
    if (args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path)):
        logger.info("Constructing model from pretrained checkpoint.")
        dmp = DMPRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        logger.info("Constructing model from provided config.")
        # Calculate the image condition length
        img_cond_len = (config["common"]["img_history_size"] 
                        * config["common"]["num_cameras"] 
                        * vision_encoder.num_patches)
        dmp = DMPRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                # No initial pos embed in the last grid size
                # since we've already done in ViT
                ("image", (config["common"]["img_history_size"], 
                    config["common"]["num_cameras"], 
                    -vision_encoder.num_patches)),  
            ],
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
        )