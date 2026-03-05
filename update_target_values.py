# import json
# import numpy as np
# import pandas as pd

# def update_reward_thresholds_with_median(scores_list, save_path):
#     """
#     10만 개의 점수 리스트를 받아서 10개 구간(Bin)으로 나누고,
#     각 구간의 커트라인(Threshold)과 중앙값(Median)을 계산하여 JSON으로 저장합니다.
#     """
#     # 1. 점수를 오름차순으로 정렬
#     scores = np.sort(np.array(scores_list))
#     n = len(scores)
    
#     results = {}
    
#     # 2. 10개의 구간으로 나누어 계산
#     # Bin 1 (하위 10%, 0~10%) ~ Bin 10 (상위 10%, 90~100%)
#     for i in range(1, 11):
#         lower_percentile = (i - 1) * 10
#         upper_percentile = i * 10
        
#         # 해당 Bin에 속하는 데이터의 인덱스 추출
#         lower_idx = int(n * (lower_percentile / 100.0))
#         upper_idx = int(n * (upper_percentile / 100.0))
#         if i == 10:  # 마지막 구간은 끝까지 포함
#             upper_idx = n
            
#         # 해당 Bin의 점수들 (Slice)
#         bin_scores = scores[lower_idx:upper_idx]
        
#         # 중앙값 및 커트라인(최소값) 계산
#         median_val = np.median(bin_scores)
#         threshold_val = np.min(bin_scores) if len(bin_scores) > 0 else 0.0
        
#         # 키 매핑 (Bin 10 -> top_10_percent, Bin 9 -> top_20_percent ...)
#         top_percent_key = (11 - i) * 10
        
#         results[f"top_{top_percent_key}_percent"] = float(threshold_val)
#         results[f"bin_{i}_median"] = float(median_val)

#     # 3. JSON 파일로 저장
#     with open(save_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4)
        
#     print(f"[*] 성공적으로 JSON 파일이 업데이트 되었습니다: {save_path}\n")
#     print("-" * 50)
#     for i in range(10, 0, -1):
#         pct = (11 - i) * 10
#         print(f"Top {pct:3d}% (Bin {i:2d}): 커트라인 = {results[f'top_{pct}_percent']:7.4f}, 중앙값(Median) = {results[f'bin_{i}_median']:7.4f}")
#     print("-" * 50)

# if __name__ == "__main__":
#     # =====================================================================
#     # [수정할 부분] 용현님의 실제 원본 점수 데이터를 불러오는 코드를 작성해주세요.
#     # 예시 1: CSV 파일에서 불러올 경우
#     # df = pd.read_csv("/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/fma_scores.csv")
#     # all_scores = df['reward_score'].tolist()
#     # 
#     # 예시 2: 임시로 테스트해볼 경우 (무작위 정규분포 점수 생성)
#     # all_scores = np.random.normal(loc=-1.0, scale=2.5, size=100000).tolist()
#     # =====================================================================
    
#     # 임시 테스트용 데이터 (실제 데이터 로드 코드로 교체하세요!)
#     print("데이터를 불러오는 중입니다...")
#     all_scores = np.random.normal(loc=-1.0, scale=2.5, size=100000).tolist()
    
#     # JSON이 저장될 경로 설정
#     json_save_path = "/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/reward_thresholds.json"
    
#     update_reward_thresholds_with_median(all_scores, json_save_path)

import pandas as pd
import numpy as np
import json

def finalize_thresholds(csv_path, json_path):
    # 1. 검증된 스코어 데이터 로드
    df = pd.read_csv(csv_path)
    scores = np.sort(df['new_score'].values) # 오름차순 정렬
    n = len(scores)
    
    results = {}
    
    print(f"[*] 총 {n}개의 데이터를 10개 구간으로 분석 중...")

    # 2. 10개 구간(Bin 1 ~ 10) 분석
    for i in range(1, 11):
        # 인덱스 계산 (0~10%, 10~20%, ..., 90~100%)
        lower_idx = int(n * (i - 1) / 10.0)
        upper_idx = int(n * i / 10.0)
        if i == 10: upper_idx = n
        
        bin_data = scores[lower_idx:upper_idx]
        
        # 통계량 추출
        threshold = np.min(bin_data) # 해당 구간의 하한선
        median = np.median(bin_data) # 해당 구간의 질량 중심(Median)
        
        # JSON 키 설정 (훈련용 top_X_percent / 평가용 bin_X_median)
        # Bin 10이 상위 10%이므로 역순 매핑
        top_pct = (11 - i) * 10
        
        results[f"top_{top_pct}_percent"] = float(threshold)
        results[f"bin_{i}_median"] = float(median)
        
    # 3. JSON 저장
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n[✅] JSON 업데이트 완료: {json_path}")
    print("-" * 60)
    print(f"{'Bin (Rank)':<12} | {'Percentile':<12} | {'Threshold':<12} | {'Median (Target)':<12}")
    print("-" * 60)
    for i in range(10, 0, -1):
        pct = (11 - i) * 10
        t = results[f"top_{pct}_percent"]
        m = results[f"bin_{i}_median"]
        print(f"Bin {i:2d} {'':<7} | Top {pct:>2d}% {'':<5} | {t:>12.4f} | {m:>15.4f}")
    print("-" * 60)

if __name__ == "__main__":
    csv_input = "/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/fma_verified_scores.csv"
    json_output = "/home/yonghyun/music-ranknet/data/processed/FMA_Scoring/reward_thresholds.json"
    finalize_thresholds(csv_input, json_output)