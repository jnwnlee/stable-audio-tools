import json
import os
import sys

import laion_clap
import pytorch_lightning as pl
import torch
import torchaudio
import transformers.modeling_utils
import transformers.utils.import_utils
from transformers import AutoModel, RobertaTokenizer, Wav2Vec2FeatureExtractor

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None


class RewardMonitorCallback(pl.Callback):
    def __init__(
        self,
        reward_model_path,
        clap_ckpt_path,
        thresholds_path,
        val_dl,
        num_samples=10,
        use_score_conditioning=False,
        music_ranknet_root=None,
        null_condition_value=-999.0,
    ):
        super().__init__()
        self.reward_model_path = reward_model_path
        self.clap_ckpt_path = clap_ckpt_path
        self.thresholds_path = thresholds_path
        self.val_dl = val_dl
        self.num_samples = num_samples
        self.use_score_conditioning = use_score_conditioning
        self.music_ranknet_root = music_ranknet_root
        self.null_condition_value = null_condition_value

        self.reward_model = None
        self.mert_processor = None
        self.mert_model = None
        self.clap_model = None
        self.target_score_list = []

    def setup(self, trainer, pl_module, stage):
        device = pl_module.device
        print(f"[*] Setting up Reward Monitor on {device}...")

        self.use_score_conditioning = True
        self.target_score_list = self._load_threshold_targets()
        if not self.target_score_list:
            print(f"!!! FATAL: Thresholds file NOT FOUND at {self.thresholds_path}")
            self.use_score_conditioning = False

        if self.music_ranknet_root and self.music_ranknet_root not in sys.path:
            sys.path.append(self.music_ranknet_root)
        from models.music_ranknet import MusicRankNet

        self.reward_model = MusicRankNet(mode="RankNet", input_dim=2049)
        self.reward_model.load_state_dict(torch.load(self.reward_model_path, map_location=device))
        self.reward_model.eval().to(device).requires_grad_(False)

        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).eval().to(device).requires_grad_(False)

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval().to(device)
        self._load_clap_checkpoint(device)
        self.clap_model.requires_grad_(False)

    def _load_threshold_targets(self):
        if not os.path.exists(self.thresholds_path):
            return []
        with open(self.thresholds_path, "r") as f:
            thresholds_data = json.load(f)
        scores = []
        for i in range(10, 0, -1):
            scores.append(float(thresholds_data.get(f"bin_{i}_median", 0.0)))
        print(f"[*] Successfully loaded 10 Thresholds: {scores}")
        return scores

    def _load_clap_checkpoint(self, device):
        original_torch_load = torch.load

        def safe_load_wrapper(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = safe_load_wrapper
            ckpt = torch.load(self.clap_ckpt_path, map_location=device)
            if "model" in ckpt:
                ckpt = ckpt["model"]
            if "text_branch.embeddings.position_ids" in ckpt:
                del ckpt["text_branch.embeddings.position_ids"]
            self.clap_model.model.load_state_dict(ckpt, strict=False)
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        finally:
            torch.load = original_torch_load

    def get_mert_embedding(self, waveform, sr, device):
        target_sr = 24000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3 and waveform.size(1) > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        waveform = waveform.squeeze(1)

        inputs = self.mert_processor(waveform.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state.mean(dim=1)

    def get_clap_audio_embedding(self, waveform, sr):
        target_sr = 48000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3 and waveform.size(1) > 1:
            waveform = torch.mean(waveform, dim=1)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        with torch.no_grad():
            return self.clap_model.get_audio_embedding_from_data(x=waveform, use_tensor=True)

    def get_clap_text_embedding(self, texts, device):
        text_data = self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        for k in text_data:
            if text_data[k].dim() == 1:
                text_data[k] = text_data[k].unsqueeze(0)
        text_data = {k: v.to(device) for k, v in text_data.items()}
        with torch.no_grad():
            return self.clap_model.model.get_text_embedding(text_data)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.reward_model is None or self.val_dl is None:
            return
        if not trainer.is_global_zero:
            return

        device = pl_module.device
        pl_module.eval()

        target_scores_log, measured_scores_log = [], []
        bin_scores = {i: [] for i in range(11)}
        generated_count = 0
        model_sr = getattr(pl_module, "sample_rate", 44100)

        with torch.no_grad():
            for batch in self.val_dl:
                if generated_count >= self.num_samples:
                    break

                audio, metadata = batch
                current_batch_size = audio.shape[0]
                for i in range(current_batch_size):
                    if generated_count >= self.num_samples:
                        break

                    m = metadata[i] if isinstance(metadata, (list, tuple)) else metadata
                    p_val = m.get("prompt", "A music song.")
                    while isinstance(p_val, (list, tuple)):
                        p_val = p_val[0]
                    clean_p = str(p_val)
                    clean_s = 10.0

                    if self.use_score_conditioning and generated_count < 10:
                        target_val = self.null_condition_value
                    else:
                        target_val = float(self.target_score_list[generated_count % 10]) if self.use_score_conditioning else self.null_condition_value

                    single_cond_input = [{"prompt": clean_p, "seconds_total": clean_s, "continuous_score": target_val}]
                    negative_cond_input = [{"prompt": "", "seconds_total": clean_s, "continuous_score": self.null_condition_value}]

                    try:
                        gen_audio = pl_module.diffusion.generate(
                            conditioning=single_cond_input,
                            negative_conditioning=negative_cond_input,
                            steps=50,
                            cfg_scale=3.5,
                            batch_size=1,
                            sample_size=int(model_sr * clean_s),
                        )

                        save_dir = os.path.join(trainer.default_root_dir, "val_samples", f"epoch_{trainer.current_epoch}")
                        os.makedirs(save_dir, exist_ok=True)
                        audio_to_save = torch.clamp(gen_audio[0].cpu().float(), -1.0, 1.0)
                        safe_prompt = "".join(x for x in clean_p[:15] if x.isalnum() or x.isspace()).replace(" ", "")
                        save_path = os.path.join(save_dir, f"sample_{generated_count}_target_{target_val:.2f}_{safe_prompt}.wav")
                        torchaudio.save(save_path, audio_to_save, model_sr)

                        mert_emb = self.get_mert_embedding(gen_audio, model_sr, device)
                        clap_audio_emb = self.get_clap_audio_embedding(gen_audio, model_sr)
                        clap_text_emb = self.get_clap_text_embedding([clean_p], device)
                        flag_tensor = torch.ones((1, 1), dtype=torch.float32, device=device)
                        concat_feat = torch.cat([flag_tensor, clap_audio_emb, mert_emb, clap_text_emb], dim=-1)
                        score = self.reward_model(concat_feat).item()

                        if target_val != self.null_condition_value:
                            target_scores_log.append(target_val)
                            measured_scores_log.append(score)
                            bin_scores[generated_count % 10].append(score)

                        if trainer.logger and isinstance(trainer.logger, pl.loggers.WandbLogger):
                            import wandb

                            trainer.logger.experiment.log(
                                {
                                    f"val_audio/Target_{target_val:.2f}": wandb.Audio(
                                        save_path,
                                        sample_rate=model_sr,
                                        caption=f"Prompt: {clean_p} | Target: {target_val:.2f} | Score: {score:.2f}",
                                    )
                                },
                                commit=False,
                            )

                        print(f"[*] Generated Sample {generated_count} (Target: {target_val:.2f})")
                        generated_count += 1
                    except Exception as e:
                        print(f"!!! Error at sample {generated_count}: {e}")
                        generated_count += 1

        if self.use_score_conditioning and target_scores_log:
            stacked = torch.stack((torch.tensor(target_scores_log), torch.tensor(measured_scores_log)))
            corr_matrix = torch.corrcoef(stacked)
            correlation = corr_matrix[0, 1].item() if not torch.isnan(corr_matrix[0, 1]) else 0.0
            pl_module.log("val/reward_correlation", correlation, rank_zero_only=True)
            print(f"\n[*] Epoch {trainer.current_epoch} - Correlation: {correlation:.4f}")
            print("-" * 50)

            for i in range(10):
                if len(bin_scores[i]) > 0:
                    avg_bin_score = sum(bin_scores[i]) / len(bin_scores[i])
                    target_score = self.target_score_list[i]
                    percentile = (i + 1) * 10
                    pl_module.log(f"val/measured_score_top_{percentile}_percent", avg_bin_score, rank_zero_only=True)
                    print(f"  - Target Top {percentile}% (Val: {target_score:.2f}) -> Measured Avg: {avg_bin_score:.4f}")
            print("-" * 50)
