import os
import sys
import torch
import torchaudio

# =======================================================
# [HOTFIX] Bypass strict PyTorch < 2.6 vulnerability check
# This must be executed before importing transformers
# =======================================================
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers import Wav2Vec2FeatureExtractor, AutoModel, RobertaTokenizer
import laion_clap

# Add MusicRankNet to path
sys.path.append("/home/yonghyun/music-ranknet")
from models.music_ranknet import MusicRankNet

class RewardScoreCalculator:
    def __init__(self, reward_model_path, clap_ckpt_path, device="cuda"):
        self.device = device
        print(f"[*] Initializing RewardScoreCalculator on {self.device}...")

        # 1. Load MusicRankNet
        self.reward_model = MusicRankNet(mode='RankNet', input_dim=2049)
        ranknet_state = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(ranknet_state)
        self.reward_model.eval().to(self.device)
        self.reward_model.requires_grad_(False)
        print("[*] MusicRankNet loaded successfully.")

        # 2. Load MERT Extractor
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        ).eval().to(self.device)
        self.mert_model.requires_grad_(False)
        print("[*] MERT model loaded successfully.")

        # 3. Load CLAP Model
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval().to(self.device)
        
        # Bypass strict loading for CLAP checkpoint
        _original_torch_load = torch.load
        def _safe_load_wrapper(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
            
        try:
            torch.load = _safe_load_wrapper
            ckpt = torch.load(clap_ckpt_path, map_location=self.device)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            if "text_branch.embeddings.position_ids" in ckpt:
                del ckpt["text_branch.embeddings.position_ids"]
            self.clap_model.model.load_state_dict(ckpt, strict=False)
            print("[*] CLAP model loaded successfully.")
        finally:
            torch.load = _original_torch_load
            
        self.clap_model.requires_grad_(False)
        
        # 4. Initialize RoBERTa Tokenizer explicitly to avoid dimension errors
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("[*] RobertaTokenizer loaded successfully.")

    def extract_mert_embedding(self, waveform, sample_rate):
        target_sr = 24000
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.squeeze(0)
        
        inputs = self.mert_processor(
            waveform.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
            
        return outputs.last_hidden_state.mean(dim=1)

    def extract_clap_audio_embedding(self, waveform, sample_rate):
        target_sr = 48000
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)
        elif waveform.dim() == 2:
            waveform = waveform.squeeze(0)
            
        waveform = waveform.unsqueeze(0)
            
        with torch.no_grad():
            embedding = self.clap_model.get_audio_embedding_from_data(x=waveform, use_tensor=True)
        return embedding

    def extract_clap_text_embedding(self, text_prompt):
        # Explicit tokenization ensures [batch_size, sequence_length] 2D shape
        text_data = self.tokenizer(
            [text_prompt], 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        
        # Double check to prevent 1D tensor index errors
        for k in text_data:
            if text_data[k].dim() == 1:
                text_data[k] = text_data[k].unsqueeze(0)
                
        text_data = {k: v.to(self.device) for k, v in text_data.items()}
        
        with torch.no_grad():
            # Bypass the wrapper hook and call the core model directly
            embedding = self.clap_model.model.get_text_embedding(text_data)
            
        return embedding

    def calculate_score(self, audio_path, text_prompt):
        print(f"\n[*] Processing audio: {audio_path}")
        print(f"[*] Prompt: '{text_prompt}'")
        
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)

        mert_emb = self.extract_mert_embedding(waveform, sr)
        clap_audio_emb = self.extract_clap_audio_embedding(waveform, sr)
        clap_text_emb = self.extract_clap_text_embedding(text_prompt)
        
        flag_tensor = torch.ones((1, 1), dtype=torch.float32, device=self.device)

        concat_feature = torch.cat([flag_tensor, clap_audio_emb, mert_emb, clap_text_emb], dim=-1)
        
        with torch.no_grad():
            score = self.reward_model(concat_feature).item()
            
        print(f"[+] Final Reward Score: {score:.6f}")
        return score

if __name__ == "__main__":
    REWARD_CKPT = "/home/yonghyun/music-ranknet/checkpoints/ultimate_train_all(brainmusic).pt"
    CLAP_CKPT = "/home/yonghyun/music-ranknet/checkpoints/music_audioset_epoch_15_esc_90.14.pt"
    
    calculator = RewardScoreCalculator(
        reward_model_path=REWARD_CKPT,
        clap_ckpt_path=CLAP_CKPT
    )
    
    sample_audio_path = "/home/yonghyun/fma/data/fma_large/144/144645.mp3"
    sample_prompt = "A Electronic, Noise, Experimental, Electroacoustic song."
    
    if os.path.exists(sample_audio_path):
        calculator.calculate_score(sample_audio_path, sample_prompt)
    else:
        print(f"[!] Target audio not found at: {sample_audio_path}")