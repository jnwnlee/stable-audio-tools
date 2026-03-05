# /home/yonghyun/stable-audio-tools/prepare_model_config.py
import json
import os

def main():
    base_config_path = "/home/yonghyun/stable-audio-tools/checkpoints/sao_small/model_config.json"
    new_config_path = "/home/yonghyun/stable-audio-tools/checkpoints/sao_small/model_config_with_score.json"

    with open(base_config_path, 'r') as f:
        config = json.load(f)

    conditioning_configs = config.get("conditioning", {}).get("configs", [])
    
    has_reward_score = any(c.get("id") == "reward_score" for c in conditioning_configs)
    
    if not has_reward_score:
        conditioning_configs.append({
            "id": "reward_score",
            "type": "number",
            "min_val": -10.0,
            "max_val": 10.0
        })
        print("[*] Successfully added 'reward_score' to conditioning configs.")
    else:
        print("[*] 'reward_score' already exists in the config.")

    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"[*] New config saved to: {new_config_path}")

if __name__ == "__main__":
    main()