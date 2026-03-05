# ultimate_core_patch.py
import re
import os

# 수정할 stable-audio-tools의 코어 파일 경로
file_path = "/home/yonghyun/stable-audio-tools/stable_audio_tools/models/conditioners.py"

with open(file_path, "r") as f:
    content = f.read()

# 이미 패치되었는지 확인
if "[ULTIMATE CORE PATCH]" in content:
    print("[*] Core module is already patched. You are good to go.")
    exit()

# 수정할 목표 지점: MultiConditioner 클래스의 forward 함수 첫 부분
target_string = "def forward(self, batch_metadata: typing.Dict[str, typing.Any], device: torch.device) -> typing.Dict[str, typing.Any]:"

# 주입할 무적의 방어 코드
injection_code = """def forward(self, batch_metadata: typing.Dict[str, typing.Any], device: torch.device) -> typing.Dict[str, typing.Any]:
        # =====================================================================
        # [ULTIMATE CORE PATCH] - 강제 딕셔너리 병합 및 JSON 메타데이터 주입
        # =====================================================================
        import json
        import os
        import torch
        
        # 1. Dataloader가 던진 튜플(망가진 형태)을 단일 딕셔너리로 강제 변환
        if isinstance(batch_metadata, (list, tuple)) and len(batch_metadata) > 0 and isinstance(batch_metadata[0], dict):
            merged = {}
            for k in batch_metadata[0].keys():
                merged[k] = [m.get(k) for m in batch_metadata]
            batch_metadata = merged
            
        if not isinstance(batch_metadata, dict):
            batch_metadata = {}
            
        # 2. 배치 사이즈 확인
        batch_size = 1
        for v in batch_metadata.values():
            if isinstance(v, (list, tuple)): batch_size = len(v); break
            if isinstance(v, torch.Tensor): batch_size = v.shape[0]; break
            
        # 3. 경로(path)를 기반으로 무조건 JSON 파일을 열어서 데이터 강제 추출
        paths = batch_metadata.get("path", [])
        if not isinstance(paths, (list, tuple)):
            paths = [paths] * batch_size
            
        prompts = []
        scores = []
        for i in range(batch_size):
            p = "A music song."
            s = 0.0
            if i < len(paths) and isinstance(paths[i], str):
                json_path = os.path.splitext(paths[i])[0] + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            jdata = json.load(jf)
                            p = jdata.get("prompt", jdata.get("text", p))
                            s = float(jdata.get("reward_score", s))
                    except Exception: pass
            prompts.append(p)
            scores.append(s)
            
        # 4. 모델이 찾는 키("prompt", "text")를 무조건 생성하여 에러 원천 차단
        batch_metadata["prompt"] = prompts
        batch_metadata["text"] = prompts
        
        # 5. Reward Score 텐서 강제 할당
        if "reward_score" not in batch_metadata:
            batch_metadata["reward_score"] = torch.tensor(scores, dtype=torch.float32, device=device)
        else:
            val = batch_metadata["reward_score"]
            if not isinstance(val, torch.Tensor):
                batch_metadata["reward_score"] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                batch_metadata["reward_score"] = val.to(device)
        # =====================================================================
"""

# 파일 내용 교체
new_content = content.replace(target_string, injection_code)

# 혹시 원본 파일에 typing 임포트가 명시되어 있지 않은 경우를 대비
if "import typing" not in new_content:
    new_content = "import typing\n" + new_content

with open(file_path, "w") as f:
    f.write(new_content)

print("[*] SUCCESSFULLY PATCHED stable_audio_tools/models/conditioners.py!")