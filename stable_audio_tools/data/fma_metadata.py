# /home/yonghyun/stable-audio-tools/stable_audio_tools/data/fma_metadata.py
import os
import json

def get_custom_metadata(info, audio):
    audio_path = info.get("path", "")
    json_path = os.path.splitext(audio_path)[0] + ".json"
    
    metadata = {
        "prompt": "",
        "text": "",
        "reward_score": 0.0
    }
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                text_val = data.get("prompt", data.get("text", "A music song."))
                
                metadata["prompt"] = text_val
                metadata["text"] = text_val
                
                metadata["reward_score"] = float(data.get("reward_score", 0.0))
            except json.JSONDecodeError:
                pass
                
    return metadata