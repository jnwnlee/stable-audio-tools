# stable-audio-tools/finetune.py
"""
=============================================================================
File: finetune.py
Description: Main fine-tuning script for Case 3 (FMA Scored Dataset) 
             with Steerability control via Classifier-Free Guidance (CFG).

Key Implementations & Fixes:
1. ScoreBinningDatasetWrapper: 
   - Converts continuous reward scores into discrete bins (1-10).
   - Implements 15% CFG dropout by assigning Bin 0 (Null/Baseline condition) 
     during training to enable conditional vs. unconditional contrast.
2. Sniper Freezing Strategy (Partial Unfreezing):
   - Freezes the entire DiT backbone to preserve pre-trained audio manifold.
   - Selectively unfreezes `score_bin` embeddings, `to_global_embed` (gateways), 
     and `adaLN` (receivers) to inject the quality conditioning signal.
3. DDP-Safe RewardMonitorCallback:
   - Evaluates real-time steerability (Target vs. Measured Score Correlation).
   - [FIX] Applied `trainer.is_global_zero` and `rank_zero_only=True` to prevent 
     race conditions and deadlocks in multi-GPU (DDP) environments.
4. CFG Generation Fix:
   - Explicitly injects `negative_conditioning` (Bin 0) during the validation
     generation step to activate the CFG mechanism.
=============================================================================
"""

import torch
import json
import os
import sys
import torchaudio
import pytorch_lightning as pl

from transformers import Wav2Vec2FeatureExtractor, AutoModel, RobertaTokenizer
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
import laion_clap

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config

class ScoreBinningDatasetWrapper(torch.utils.data.Dataset):
    """
    Intercepts the dataloader to convert continuous scalars to discrete bins (1-10).
    Applies CFG dropout by assigning class 0 with a given probability during training.
    """
    def __init__(self, dataset, thresholds, is_training=True, cfg_drop_rate=0.15):
        self.dataset = dataset
        self.thresholds = sorted(thresholds) # Ensure ascending order
        self.is_training = is_training
        self.cfg_drop_rate = cfg_drop_rate

    def __len__(self):
        return len(self.dataset)
        
    def get_bin_index(self, score: float) -> int:
        if score < self.thresholds[0]: return 1
        if score >= self.thresholds[-1]: return 10
        
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= score < self.thresholds[i+1]:
                return i + 1 
        return 1

    def __getitem__(self, idx):
        audio, metadata = self.dataset[idx]

        # [추가] 원본 데이터 캐시 오염(영구 0번 Bin 고정) 방지를 위한 깊은 복사
        metadata = metadata.copy()

        raw_score = metadata.get('reward_score', 0.0)
        if torch.is_tensor(raw_score): raw_score = raw_score.item()
        elif isinstance(raw_score, list): raw_score = raw_score[0]
        
        bin_idx = self.get_bin_index(float(raw_score))
        
        # Classifier-Free Guidance (CFG) Score Drop logic
        if self.is_training and torch.rand(1).item() < self.cfg_drop_rate:
            bin_idx = 0 # 0 is the unconditional/null class
            
        metadata['score_bin'] = bin_idx
        return audio, metadata

# ---------------------------------------------------------
# 1. Custom Reward Model Imports
# ---------------------------------------------------------
sys.path.append("/home/yonghyun/music-ranknet")
from models.music_ranknet import MusicRankNet

# ---------------------------------------------------------
# 2. Callbacks
# ---------------------------------------------------------
class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

class RewardMonitorCallback(pl.Callback):
    def __init__(self, reward_model_path, clap_ckpt_path, thresholds_path, val_dl, num_samples=10, use_score_conditioning=False):
        super().__init__()
        self.reward_model_path = reward_model_path
        self.clap_ckpt_path = clap_ckpt_path
        self.thresholds_path = thresholds_path
        self.val_dl = val_dl
        self.num_samples = num_samples 
        self.use_score_conditioning = use_score_conditioning
        
        self.reward_model = None
        self.mert_processor = None
        self.mert_model = None
        self.clap_model = None
        self.target_score_list = []

    def setup(self, trainer, pl_module, stage):
        device = pl_module.device
        print(f"[*] Setting up Reward Monitor on {device}...")
        
        self.use_score_conditioning = True 
        
        if os.path.exists(self.thresholds_path):
            with open(self.thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
            
            self.target_score_list = []
            for i in range(10, 0, -1): # From Bin 10 to Bin 1
                key = f"bin_{i}_median"
                val = thresholds_data.get(key, 0.0)
                self.target_score_list.append(float(val))
                
            print(f"[*] Successfully loaded 10 Thresholds: {self.target_score_list}")
        else:
            print(f"!!! FATAL: Thresholds file NOT FOUND at {self.thresholds_path}")
            self.use_score_conditioning = False

        self.reward_model = MusicRankNet(mode='RankNet', input_dim=2049) 
        self.reward_model.load_state_dict(torch.load(self.reward_model_path, map_location=device))
        self.reward_model.eval().to(device).requires_grad_(False)
        
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).eval().to(device).requires_grad_(False)

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval().to(device)
        
        _original_torch_load = torch.load
        def _safe_load_wrapper(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
            
        try:
            torch.load = _safe_load_wrapper
            ckpt = torch.load(self.clap_ckpt_path, map_location=device)
            if 'model' in ckpt: ckpt = ckpt['model']
            if "text_branch.embeddings.position_ids" in ckpt: del ckpt["text_branch.embeddings.position_ids"]
            self.clap_model.model.load_state_dict(ckpt, strict=False)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        finally:
            torch.load = _original_torch_load
            
        self.clap_model.requires_grad_(False)

    def get_mert_embedding(self, waveform, sr, device):
        target_sr = 24000
        if sr != target_sr: waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3 and waveform.size(1) > 1: waveform = torch.mean(waveform, dim=1, keepdim=True)
        waveform = waveform.squeeze(1) 
        
        inputs = self.mert_processor(waveform.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.mert_model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state.mean(dim=1)

    def get_clap_audio_embedding(self, waveform, sr):
        target_sr = 48000
        if sr != target_sr: waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3 and waveform.size(1) > 1: waveform = torch.mean(waveform, dim=1)
        elif waveform.dim() == 3: waveform = waveform.squeeze(1)
            
        with torch.no_grad():
            return self.clap_model.get_audio_embedding_from_data(x=waveform, use_tensor=True)

    def get_clap_text_embedding(self, texts, device):
        text_data = self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        for k in text_data:
            if text_data[k].dim() == 1: text_data[k] = text_data[k].unsqueeze(0)
        text_data = {k: v.to(device) for k, v in text_data.items()}
        with torch.no_grad():
            return self.clap_model.model.get_text_embedding(text_data)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.reward_model is None or self.val_dl is None: return

        # DDP 환경에서 충돌을 막기 위해 0번 GPU(대장)만 실행하도록 차단
        if not trainer.is_global_zero:
            return
        
        device = pl_module.device
        pl_module.eval()

        target_scores_log, measured_scores_log = [], []
        bin_scores = {i: [] for i in range(11)} # Include Bin 0
        generated_count = 0
        model_sr = getattr(pl_module, 'sample_rate', 44100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dl):
                if generated_count >= self.num_samples: break
                
                audio, metadata = batch
                current_batch_size = audio.shape[0] 

                for i in range(current_batch_size):
                    if generated_count >= self.num_samples: break
                    
                    m = metadata[i] if isinstance(metadata, (list, tuple)) else metadata
                    
                    # 1. Text and basic settings
                    p_val = m.get('prompt', "A music song.")
                    while isinstance(p_val, (list, tuple)): p_val = p_val[0]
                    clean_p = str(p_val)
                    clean_s = 10.0 

                    # Bin 0 (Baseline) vs Bin 1~10 (Steering) Logic
                    if self.use_score_conditioning and generated_count < 10:
                        target_bin = 0
                        target_val = 0.0 # Baseline will be excluded from statistical calculations
                    else:
                        target_bin = 10 - (generated_count % 10) if self.use_score_conditioning else 0
                        target_val = float(self.target_score_list[generated_count % 10]) if self.use_score_conditioning else 0.0

                    single_cond_input = [{
                        "prompt": clean_p,
                        "seconds_total": clean_s,
                        "score_bin": target_bin
                    }]

                    # [추가] CFG 상쇄를 막기 위한 완벽한 Null 조건 명시
                    negative_cond_input = [{
                        "prompt": "",              # 프롬프트 비우기
                        "seconds_total": clean_s,  # 길이는 유지
                        "score_bin": 0             # 점수는 Null(0)로 
                    }]
                    
                    try:
                        target_sample_size = int(model_sr * clean_s)
                        
                        # # Generate audio
                        # gen_audio = pl_module.diffusion.generate(
                        #     conditioning=single_cond_input,
                        #     steps=50,
                        #     cfg_scale=1.5, 
                        #     batch_size=1,
                        #     sample_size=target_sample_size
                        # )
                        # Generate audio
                        gen_audio = pl_module.diffusion.generate(
                            conditioning=single_cond_input,
                            negative_conditioning=negative_cond_input, # CFG 상쇄 방지를 위한 Null 조건 추가
                            steps=50,
                            cfg_scale=1.5, 
                            batch_size=1,
                            sample_size=target_sample_size
                        )
                        
                        # Save locally
                        save_dir = os.path.join(trainer.default_root_dir, "val_samples", f"epoch_{trainer.current_epoch}")
                        os.makedirs(save_dir, exist_ok=True)
                        audio_to_save = torch.clamp(gen_audio[0].cpu().float(), -1.0, 1.0)
                        
                        safe_prompt = "".join(x for x in clean_p[:15] if x.isalnum() or x.isspace()).replace(" ", "")
                        filename = f"sample_{generated_count}_bin_{target_bin}_{safe_prompt}.wav"
                        save_path = os.path.join(save_dir, filename)
                        torchaudio.save(save_path, audio_to_save, model_sr)

                        # Evaluate with reward model
                        mert_emb = self.get_mert_embedding(gen_audio, model_sr, device)
                        clap_audio_emb = self.get_clap_audio_embedding(gen_audio, model_sr)
                        clap_text_emb = self.get_clap_text_embedding([clean_p], device)
                        flag_tensor = torch.ones((1, 1), dtype=torch.float32, device=device)
                        concat_feat = torch.cat([flag_tensor, clap_audio_emb, mert_emb, clap_text_emb], dim=-1)
                        
                        score = self.reward_model(concat_feat).item()

                        # Statistics and WandB Audio Logging
                        if target_bin != 0:
                            target_scores_log.append(target_val)
                            measured_scores_log.append(score)
                            bin_scores[generated_count % 10].append(score)

                        if trainer.logger and isinstance(trainer.logger, pl.loggers.WandbLogger):
                            import wandb
                            trainer.logger.experiment.log({
                                f"val_audio/Bin_{target_bin}": wandb.Audio(
                                    save_path,
                                    sample_rate=model_sr,
                                    caption=f"Prompt: {clean_p} | Bin: {target_bin} | Target: {target_val:.2f} | Score: {score:.2f}"
                                )
                            }, commit=False)

                        print(f"[*] Generated Sample {generated_count} (Bin: {target_bin})")
                        generated_count += 1 # Increment count is required!

                    except Exception as e:
                        print(f"!!! Error at sample {generated_count}: {e}")
                        generated_count += 1 # Skip to next on error
                        continue
                    
        # Final log calculation and output
        if self.use_score_conditioning and len(target_scores_log) > 0:
            stacked = torch.stack((torch.tensor(target_scores_log), torch.tensor(measured_scores_log)))
            corr_matrix = torch.corrcoef(stacked)
            correlation = corr_matrix[0, 1].item() if not torch.isnan(corr_matrix[0, 1]) else 0.0
            
            #self.log("val/reward_correlation", correlation, sync_dist=True)
            self.log("val/reward_correlation", correlation, rank_zero_only=True)
            print(f"\n[*] Epoch {trainer.current_epoch} - Correlation: {correlation:.4f}")
            print("-" * 50)
             
            for i in range(10):
                if len(bin_scores[i]) > 0:
                    avg_bin_score = sum(bin_scores[i]) / len(bin_scores[i])
                    target_score = self.target_score_list[i]
                    percentile = (i + 1) * 10
                    
                    #self.log(f"val/measured_score_top_{percentile}_percent", avg_bin_score, sync_dist=True)
                    self.log(f"val/measured_score_top_{percentile}_percent", avg_bin_score, rank_zero_only=True)
                    print(f"  - Target Top {percentile}% (Val: {target_score:.2f}) -> Measured Avg: {avg_bin_score:.4f}")
            print("-" * 50)
            
# ---------------------------------------------------------
# 3. Main Function
# ---------------------------------------------------------
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed if args.seed is not None else 42
    if os.environ.get("SLURM_PROCID") is not None: seed += int(os.environ.get("SLURM_PROCID"))
    pl.seed_everything(seed, workers=True)

    with open(args.model_config) as f: model_config = json.load(f)
    with open(args.dataset_config) as f: dataset_config = json.load(f)

    has_score_cond = any(cond.get("id") == "score_bin" for cond in model_config.get("conditioning", {}).get("configs", []))

    temp_train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )
    train_set = temp_train_dl.dataset # Extract dataset!

    thresholds_path = "/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/reward_thresholds.json"
    with open(thresholds_path, 'r') as f:
        th_data = json.load(f)
    target_score_list = [float(th_data.get(f"top_{i*10}_percent", 0.0)) for i in range(1, 11)]

    # Create wrapped Train Dataset
    wrapped_train_set = ScoreBinningDatasetWrapper(
        train_set, 
        thresholds=target_score_list, 
        is_training=True, 
        cfg_drop_rate=0.15
    )

    # Final assembly of Train DataLoader (keep original collate_fn)
    train_dl = torch.utils.data.DataLoader(
        wrapped_train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, 
        pin_memory=True,
        collate_fn=getattr(temp_train_dl, 'collate_fn', None)
    )

    val_dl = None
    if args.val_dataset_config:
        with open(args.val_dataset_config) as f: val_dataset_config = json.load(f)
        
        # Extract and wrap Validation similarly
        temp_val_dl = create_dataloader_from_config(
            val_dataset_config, batch_size=args.batch_size, num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"], audio_channels=model_config.get("audio_channels", 2), shuffle=False
        )
        val_set = temp_val_dl.dataset
        
        wrapped_val_set = ScoreBinningDatasetWrapper(
            val_set, 
            thresholds=target_score_list, 
            is_training=False, 
            cfg_drop_rate=0.0
        )
        
        val_dl = torch.utils.data.DataLoader(
            wrapped_val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=getattr(temp_val_dl, 'collate_fn', None)
        )

    model = create_model_from_config(model_config)
    if args.pretrained_ckpt_path: copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    # print("[*] 🥶 Freezing DiT core backbone to protect audio quality...")
    # model.requires_grad_(False) # 전체를 완벽하게 얼립니다.
    
    # # 오직 아래 3개의 텐서만 정확하게 이름을 매칭하여 학습을 허용합니다.
    # # 1. Score binning을 담당하는 임베딩 레이어 (score_bin)
    # # 2 & 3. Score binning을 모델 내부에 전달 (Global Embedder)
    target_tensors = [
        "conditioner.conditioners.score_bin.embedding.weight",
        "model.model.to_global_embed.0.weight",
        "model.model.to_global_embed.2.weight"
    ]

    print("\n[*] --- 🎯 SNIPER AUDIT: List of UNFROZEN Parameters ---")
    
    # # ==========================================================
    # # 1. AdaLN 안테나 개방
    # # ==========================================================
    # print("[*] 🥶 Freezing DiT core backbone, but opening AdaLN gates...")
    # model.requires_grad_(False)
    
    # # 조향 신호가 흐르는 핵심 통로인 'adaLN' 파라미터를 추가로 해제합니다.
    # print("\n[*] --- List of UNFROZEN Parameters ---")
    
    
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # score_bin(임베딩), to_global_embed(관문), adaLN(수신부)를 모두 엽니다.
        if any(key in name for key in ["score_bin", "to_global_embed", "adaLN"]):
            param.requires_grad_(True)
            unfrozen_count += 1
            print(f"  🟢 [UNFROZEN] {name} (Shape: {list(param.shape)})")
            
    print(f"[*] --- Total {unfrozen_count} tensors successfully unfrozen ---\n")
    
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if name in target_tensors:
            param.requires_grad_(True)
            unfrozen_count += 1
            print(f"  🟢 [UNFROZEN] {name} (Shape: {list(param.shape)})")
            
    print(f"[*] --- Total {unfrozen_count} tensors successfully unfrozen ---\n")
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    def custom_configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-3)
        return optimizer

    import types
    training_wrapper.configure_optimizers = types.MethodType(custom_configure_optimizers, training_wrapper)
    print("[*] ⚡ Applied custom high-LR optimizer for ScoreBin embeddings.")
    
    logger = None
    checkpoint_dir = args.save_dir if args.save_dir else "checkpoints"

    if args.logger == 'wandb':
        WANDB_PROJECT = "music-steerability-study"
        run_name = args.name
        
        logger = pl.loggers.WandbLogger(
            project=WANDB_PROJECT, 
            name=run_name,
            group="Case3_FMA_Scored"
        )
        logger.watch(training_wrapper)
        
        if args.save_dir:
            checkpoint_dir = os.path.join(args.save_dir, WANDB_PROJECT, run_name, "checkpoints")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, 
        dirpath=checkpoint_dir, 
        save_top_k=-1
    )
    
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    reward_ckpt_path = "/home/yonghyun/music-ranknet/checkpoints/ultimate_train_all(brainmusic).pt"
    clap_ckpt_path = "/home/yonghyun/music-ranknet/checkpoints/music_audioset_epoch_15_esc_90.14.pt" 
    thresholds_path = "/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/reward_thresholds.json"
    
    reward_callback = RewardMonitorCallback(
        reward_model_path=reward_ckpt_path, clap_ckpt_path=clap_ckpt_path, thresholds_path=thresholds_path,
        val_dl=val_dl if val_dl else train_dl, num_samples=100, use_score_conditioning=has_score_cond
    )

    args_dict = vars(args)
    args_dict.update({"model_config": model_config, "dataset_config": dataset_config})
    if args.logger == 'wandb': push_wandb_config(logger, args_dict)

    strategy = 'ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else "auto"
    val_args = {"check_val_every_n_epoch": None, "val_check_interval": args.val_every} if args.val_every > 0 else {}

    trainer = pl.Trainer(
        devices="auto", 
        accelerator="gpu", 
        num_nodes=args.num_nodes, 
        strategy=strategy, 
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        limit_val_batches=100, 
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback, reward_callback],
        logger=logger, 
        log_every_n_steps=1, 
        max_epochs=10000000, 
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val, 
        reload_dataloaders_every_n_epochs=0, 
        num_sanity_val_steps=1, 
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
    
"""
CUDA_VISIBLE_DEVICES=8,9 PYTHONPATH=. python3 ./finetune.py  \
    --dataset-config ./configs/dataset_fma_scored.json  \
    --val-dataset-config ./configs/dataset_fma_scored.json  \
    --model-config ./checkpoints/sao_small/model_config_with_score.json  \
    --pretrained-ckpt-path ./checkpoints/sao_small/model.safetensors \
    --name "sao_small_case3_${NOW}" \
    --save-dir ./results/sao_small_case3  \
    --batch-size 2  \
    --accum-batches 4 \
    --precision 16-mixed  \
    --checkpoint-every 1000  \
    --val-every 1000
"""