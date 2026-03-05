import os
import json

def get_custom_metadata(info, audio_path):
    # Logic: Look for a .json file with the same name as the audio file
    json_path = os.path.splitext(audio_path)[0] + ".json"
    
    metadata = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            # This 'reward_score' key must match the conditioner name in model_config
            metadata["reward_score"] = data.get("score", 0.5)
    else:
        metadata["reward_score"] = 0.5 # Default fallback
        
    return metadata