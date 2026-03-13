# stable-audio-tools
Training and inference code for audio generation models

# Install

The library can be installed from PyPI with:
```bash
$ pip install stable-audio-tools
```

To run the training scripts or inference code, you'll want to clone this repository, navigate to the root, and run:
```bash
$ pip install .
```

# Requirements
Requires PyTorch 2.5 or later for Flash Attention and Flex Attention support

Development for the repo is done in Python 3.10

# Interface

A basic Gradio interface is provided to test out trained models. 

For example, to create an interface for the [`stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0) model, once you've accepted the terms for the model on Hugging Face, you can run:
```bash
$ python3 ./run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0
```

The `run_gradio.py` script accepts the following command line arguments:

- `--pretrained-name`
  - Hugging Face repository name for a Stable Audio Tools model
  - Will prioritize `model.safetensors` over `model.ckpt` in the repo
  - Optional, used in place of `model-config` and `ckpt-path` when using pre-trained model checkpoints on Hugging Face
- `--model-config`
  - Path to the model config file for a local model
- `--ckpt-path`
  - Path to unwrapped model checkpoint file for a local model
- `--pretransform-ckpt-path` 
  - Path to an unwrapped pretransform checkpoint, replaces the pretransform in the model, useful for testing out fine-tuned decoders
  - Optional
- `--share`
  - If true, a publicly shareable link will be created for the Gradio demo
  - Optional
- `--username` and `--password`
  - Used together to set a login for the Gradio demo
  - Optional
- `--model-half`
  - If true, the model weights to half-precision
  - Optional

# Training

## Prerequisites
Before starting your training run, you'll need a model config file, as well as a dataset config file. For more information about those, refer to the Configurations section below

The training code also requires a Weights & Biases account to log the training outputs and demos. Create an account and log in with:
```bash
$ wandb login
```

## Start training
To start a training run, run the `train.py` script in the repo root with:
```bash
$ python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name harmonai_train
```

The `--name` parameter will set the project name for your Weights and Biases run.

## Training wrappers and model unwrapping
`stable-audio-tools` uses PyTorch Lightning to facilitate multi-GPU and multi-node training. 

When a model is being trained, it is wrapped in a "training wrapper", which is a `pl.LightningModule` that contains all of the relevant objects needed only for training. That includes things like discriminators for autoencoders, EMA copies of models, and all of the optimizer states.

The checkpoint files created during training include this training wrapper, which greatly increases the size of the checkpoint file.

`unwrap_model.py` in the repo root will take in a wrapped model checkpoint and save a new checkpoint file including only the model itself.

That can be run with from the repo root with:
```bash
$ python3 ./unwrap_model.py --model-config /path/to/model/config --ckpt-path /path/to/wrapped/ckpt --name model_unwrap
```

Unwrapped model checkpoints are required for:
  - Inference scripts
  - Using a model as a pretransform for another model (e.g. using an autoencoder model for latent diffusion)
  - Fine-tuning a pre-trained model with a modified configuration (i.e. partial initialization)

## Fine-tuning
Fine-tuning a model involves continuning a training run from a pre-trained checkpoint. 

To continue a training run from a wrapped model checkpoint, you can pass in the checkpoint path to `train.py` with the `--ckpt-path` flag.

To start a fresh training run using a pre-trained unwrapped model, you can pass in the unwrapped checkpoint to `train.py` with the `--pretrained-ckpt-path` flag.

## Research Fine-Tuning (Continuous Score Conditioning)

If you are running steerability experiments with continuous reward scores, use `finetune.py` with:
- a model config that defines `continuous_score` conditioning
- an audio dataset where each audio file has a sidecar JSON containing `prompt`/`text` and `reward_score`

Recommended environment variables for reproducible research runs:

```bash
export CUDA_VISIBLE_DEVICES=8,9
export MUSIC_RANKNET_ROOT=/path/to/music-ranknet
export REWARD_MODEL_CKPT=$MUSIC_RANKNET_ROOT/checkpoints/ultimate_train_all(brainmusic).pt
export CLAP_MODEL_CKPT=$MUSIC_RANKNET_ROOT/checkpoints/music_audioset_epoch_15_esc_90.14.pt
export REWARD_THRESHOLDS_PATH=$MUSIC_RANKNET_ROOT/data/processed/FMA_Scoring/reward_thresholds.json

# Optional scheduler overrides
export SA_WARMUP_STEPS=1000
export SA_TOTAL_STEPS=300000
export SA_NUM_CYCLES=18
```

Example launch command:

```bash
python3 ./finetune.py \
  --dataset-config ./configs/dataset_fma_scored.json \
  --val-dataset-config ./configs/dataset_fma_scored.json \
  --model-config ./checkpoints/sao_small/model_config_with_score.json \
  --pretrained-ckpt-path ./checkpoints/sao_small/model.safetensors \
  --name "sao_small_case3_run01" \
  --save-dir ./results/sao_small_case3 \
  --batch-size 2 \
  --accum-batches 4 \
  --precision 16-mixed \
  --checkpoint-every 1000 \
  --val-every 1000
```

One-command launcher (recommended):

```bash
./scripts/run_finetune_case3.sh
```

Show all available overrides:

```bash
./scripts/run_finetune_case3.sh --help
```

### Conditioning strategy options

You can switch score-conditioning behavior without editing code by setting `SA_UNFREEZE_PROFILE`:

- `hybrid` (default): unfreezes `continuous_score` + `to_global_embed` + `adaLN` + `input_add_adapter`
- `adaln`: focuses on AdaLN modulation paths
- `adapter`: focuses on additive adapter path (`input_add_adapter`)
- `global`: focuses on global embedding route (`to_global_embed`)
- `minimal`: only score-conditioning heads (most conservative)

Example:

```bash
SA_UNFREEZE_PROFILE=adaln RUN_NAME=sao_adaln_exp1 ./scripts/run_finetune_case3.sh
```

By default, checkpoints are also grouped by profile under:
`./results/sao_small_case3/<profile>/`

Advanced custom override (comma-separated name substrings):

```bash
SA_TRAINABLE_NAME_KEYS="continuous_score,adaLN,input_add_adapter" ./scripts/run_finetune_case3.sh
```

### Suggested experiment matrix

For score-conditioning research, a practical baseline matrix is:

1. `minimal` (stability baseline)
2. `adaln` (conditioning strength via modulation)
3. `adapter` (conditioning strength via additive route)
4. `hybrid` (best overall candidate in many settings)

Keep all other hyperparameters fixed while comparing:
- same prompts for validation
- same random seed
- same `cfg_scale` / diffusion steps
- same data split

Validation cost controls (no code change needed):

```bash
SA_VAL_NUM_SAMPLES=40 SA_VAL_GEN_STEPS=30 SA_VAL_CFG_SCALE=3.5 ./scripts/run_finetune_case3.sh
```

Recommended for quick smoke validation:
- `SA_VAL_NUM_SAMPLES=20`
- `SA_VAL_GEN_STEPS=20`

The reward-monitor validation also logs operational metrics:
- `val/reward_eval_time_sec`
- `val/reward_generated_count`
- `val/reward_scored_count`
- `val/reward_error_count`
- `val/reward_audio_log_count`
- `val/score_monotonicity` (pairwise ranking consistency between target and measured scores)

### Other high-value score-conditioning ideas

If you want to push performance further, these are often effective:
- **Two-stage training**: `minimal` warm-up, then continue with `hybrid`
- **Curriculum on condition dropout**: start with lower `SA_CFG_DROP_RATE`, then increase
- **Score normalization**: z-score or quantile normalization of `reward_score` before conditioning
- **Monotonicity evaluation**: track correlation + pairwise monotonic ordering, not just average score

Practical notes for researchers:
- Keep training deterministic where possible: set `--seed` and log all env vars in your run metadata.
- Verify trainable parameter subsets at startup (`finetune.py` prints all unfrozen tensors).
- Use short smoke runs first (few hundred steps) before long runs to validate dataloading, conditioning keys, and logging.
- Keep validation prompts fixed across experiments for fair steerability comparisons.

## Additional training flags

Additional optional flags for `train.py` include:
- `--config-file`
  - The path to the defaults.ini file in the repo root, required if running `train.py` from a directory other than the repo root
- `--pretransform-ckpt-path`
  - Used in various model types such as latent diffusion models to load a pre-trained autoencoder. Requires an unwrapped model checkpoint.
- `--save-dir`
  - The directory in which to save the model checkpoints
- `--checkpoint-every`
  - The number of steps between saved checkpoints.
  - *Default*: 10000
- `--batch-size`
  - Number of samples per-GPU during training. Should be set as large as your GPU VRAM will allow.
  - *Default*: 8
- `--num-gpus`
  - Number of GPUs per-node to use for training
  - *Default*: 1
- `--num-nodes`
  - Number of GPU nodes being used for training
  - *Default*: 1
- `--accum-batches`
  - Enables and sets the number of batches for gradient batch accumulation. Useful for increasing effective batch size when training on smaller GPUs.
- `--strategy`
  - Multi-GPU strategy for distributed training. Setting to `deepspeed` will enable DeepSpeed ZeRO Stage 2.
  - *Default*: `ddp` if `--num_gpus` > 1, else None
- `--precision`
  - floating-point precision to use during training
  - *Default*: 16
- `--num-workers`
  - Number of CPU workers used by the data loader
- `--seed`
  - RNG seed for PyTorch, helps with deterministic training

# Configurations
Training and inference code for `stable-audio-tools` is based around JSON configuration files that define model hyperparameters, training settings, and information about your training dataset.

## Model config
The model config file defines all of the information needed to load a model for training or inference. It also contains the training configuration needed to fine-tune a model or train from scratch.

The following properties are defined in the top level of the model configuration:

- `model_type`
  - The type of model being defined, currently limited to one of `"autoencoder", "diffusion_uncond", "diffusion_cond", "diffusion_cond_inpaint", "diffusion_autoencoder", "lm"`.
- `sample_size`
  - The length of the audio provided to the model during training, in samples. For diffusion models, this is also the raw audio sample length used for inference.
- `sample_rate`
  - The sample rate of the audio provided to the model during training, and generated during inference, in Hz.
- `audio_channels`
  - The number of channels of audio provided to the model during training, and generated during inference. Defaults to 2. Set to 1 for mono.
- `model`
  - The specific configuration for the model being defined, varies based on `model_type`
- `training`
  - The training configuration for the model, varies based on `model_type`. Provides parameters for training as well as demos.

## Dataset config
`stable-audio-tools` currently supports two kinds of data sources: local directories of audio files, and WebDataset datasets stored in Amazon S3. More information can be found in [the dataset config documentation](docs/datasets.md)

# Todo
- [ ] Add troubleshooting section
- [ ] Add contribution guidelines 
