import argparse
import os
from train.train import train

from accelerate.logging import get_logger

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Main script to Train Diffusion Mamba Policy")
    # Required File Paths
    parser.add_argument("--config_path", 
                        type=str, 
                        default="configs/config.yaml", 
                        help="Path to config file",
                        )
    parser.add_argument("--load_from_hdf5",
                        action="store_true",
                        default=False,
                        help=(
                            "Whether to load the dataset directly from HDF5 files. "
                            "If False, the dataset will be loaded using producer-consumer pattern, "
                            "where the producer reads TFRecords and saves them to buffer, and the consumer reads from buffer."
                            )
                        )
    parser.add_argument("--pretrain_vision_encoder",
                        type=str,
                        default=None,
                        help="Path to pre-trained vision encoder",
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="The output directory where the model predictions and checkpoints will be written.",
                        )
    parser.add_argument("--deepspeed",
                        type=str,
                        default=None,
                        help="Enable DeepSpeed and pass the path to its config file",
                        )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.",
                        )
    parser.add_argument("--pretrain_model", type=str, default=None,
                        help=(
                        "Path or name of a pretrained checkpoint to load the model from.\n",
                        "   This can be either:\n"
                        "   - a string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co, e.g., `robotics-diffusion-transformer/rdt-1b`,\n"
                        "   - a path to a *directory* containing model weights saved using [`~RDTRunner.save_pretrained`] method, e.g., `./my_model_directory/`.\n"
                        "   - a path to model checkpoint (*.pt), .e.g, `my_model_directory/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt`"
                        "   - `None` if you are randomly initializing model using configuration at `config_path`.")
                        )
    
    # training parameters
    parser.add_argument("--train_batch_size", type=int, default=4, help="The training batch size.")
    parser.add_argument("--sample_batch_size", type=int, default=8, help="The sample batch size.")
    parser.add_argument("--num_sample_batches", type=int, default=2, help="Number of batches to sample from the dataset.")

    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The initial learning rate.")

    parser.add_argument("--alpha", type=float, default=0.9, help="The moving average coefficient for each dataset's loss.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--sample_period", type=int, default=-1,
        help=(
            "Run sampling every X steps. During the sampling phase, the model will sample a trajectory"
            " and report the error between the sampled trajectory and groud-truth trajectory"
            " in the training batch."
        ),
    )

    # checkpoints
    parser.add_argument("--checkpointing_period",
            type=int,
            default=500,
            help=(
                "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
                "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
                "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
                "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
                "instructions."
            )
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
            help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_period`, or `"latest"` to automatically select the last available checkpoint.'
            )
    )
    parser.add_argument("--checkpoint_total_limit",
            type=int,
            default=None,
            help=(
                "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
                " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
                " for more details"
            )
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # training options
    parser.add_argument('--dataset_type', type=str, default="pretrain", required=False, help="Whether to load the pretrain dataset or finetune dataset.")
    parser.add_argument("--state_noise_snr", type=float, default=None,
        help=(
            "The signal-to-noise ratio (SNR, unit: dB) for adding noise to the states. "
            "Default is None, which means no noise is added."
        )
    )
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--image_aug", action="store_true", default=False, help="Whether or not to apply image augmentation (ColorJitter, blur, noise, etc) to the input images.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--set_grads_to_none",action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    # learning rate scheduler options
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]'
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    # optimization options  
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    # multi-gpu training
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # logging options
    parser.add_argument("--logging_dir", type=str, default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. "
            "Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        )
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help= "The integration to report the results and logs to. Supported platforms are tensorboard(default), wandb and comet_ml. Use all to report to all integrations."
    )

    # args setting
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
