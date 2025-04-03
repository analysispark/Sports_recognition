#! Projects/main.py

'''
DATE        : 2025-04-03
DIRECTOR    : jihoon, park
              taehoon, kim
VERSION     : beta v.0.1

---
python main.py --/videos/sample.mp4
---
'''

import os
import argparse
from modules.track_player_movement import track_player_movement
from modules.visualize import visualize_movement_pattern
from modules.calculates import *

def main(video_name, output_dir):
    print(f"처리할 비디오 파일: {video_name}")
    print(f"출력 폴더: {output_dir}")
    
    current_dir = os.getcwd()

    video_dir = os.path.join(current_dir, "videos/")
    video_path = os.path.join(video_dir, video_name)
    
    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다. ({video_path})")
        return
    
    print("=== 트래킹 시작 ===")
    positions, scoring_events = track_player_movement(video_path)

    if positions:
        print("\n=== 움직임 분석 시작 ===")
        visualize_movement_pattern(positions, scoring_events)
        analyze_movement(positions)

        print("\n=== 득점/실점 통계 ===")
        scores = len([e for e in scoring_events if e[0] == "득점"])
        concedes = len([e for e in scoring_events if e[0] == "실점"])
        print(f"총 득점: {scores}")
        print(f"총 실점: {concedes}")

        # 분석 결과 저장
        save_analysis_results(positions, scoring_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="경기분석 프로그램")

    # First index: 비디오 경로
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="분석할 비디오 파일 경로"
    )

    # Second index: 출력 폴더
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="분석결과지 출력 폴더 경로"
    )

    args = parser.parse_args()
    main(args.video_path, args.output_dir)
