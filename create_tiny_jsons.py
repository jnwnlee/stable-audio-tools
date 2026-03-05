# create_tiny_jsons.py
import json
import os
from tqdm import tqdm

base_audio_dir = "/home/yonghyun/fma/data/fma_large"
jsonl_path = "/home/yonghyun/stable-audio-tools/configs/metadata_fma_scored.jsonl"

print(f"[*] Reading metadata from {jsonl_path}")

if not os.path.exists(jsonl_path):
    print(f"[!] Error: {jsonl_path} file doesn't exist.")
    exit()

with open(jsonl_path, 'r') as f:
    lines = f.readlines()

success_count = 0

for line in tqdm(lines, desc="Creating tiny JSONs"):
    entry = json.loads(line)
    relpath = entry.get("relpath", "")
    
    if relpath.startswith(base_audio_dir):
        relpath = os.path.relpath(relpath, base_audio_dir)
        
    mp3_full_path = os.path.join(base_audio_dir, relpath)
    
    json_full_path = os.path.splitext(mp3_full_path)[0] + ".json"
    
    prompt_text = entry.get("prompt", entry.get("text", "A music song."))
    reward_score = entry.get("reward_score", 0.0)
    
    out_data = {
        "prompt": prompt_text,
        "reward_score": reward_score
    }
    
    os.makedirs(os.path.dirname(json_full_path), exist_ok=True)
    
    with open(json_full_path, 'w') as out_f:
        json.dump(out_data, out_f)
        
    success_count += 1

print(f"[*] Successfully created {success_count} JSON files in {base_audio_dir}")