#! Projects/modules/calculates.py

import numpy as np
import datetime

# 픽셀 좌표를 실제 미터 단위로 변환
def pixels_to_meters(
    pixel_coords,
    court_width_pixels,
    court_height_pixels,
    court_width_meters=8,
    court_height_meters=8,
):
    x_pixel, y_pixel = pixel_coords
    # 중앙을 기준으로 변환
    x_meters = (x_pixel - (court_width_pixels / 2)) * (
        court_width_meters / court_width_pixels
    )
    y_meters = ((court_height_pixels / 2) - y_pixel) * (
        court_height_meters / court_height_pixels
    )
    return (x_meters, y_meters)

def analyze_movement(positions, court_width=640, court_height=360):
    if not positions:
        print("분석할 위치 데이터가 없습니다.")
        return

    # 트래킹 중단 지점(-1, -1)을 제외한 유효한 위치 데이터만 추출
    valid_positions = [pos for pos in positions if pos != (-1, -1)]

    # 픽셀 좌표를 미터 단위로 변환
    positions_meters = [
        pixels_to_meters(pos, court_width, court_height) for pos in valid_positions
    ]

    x_coords_m, y_coords_m = zip(*positions_meters)

    # 1. 기본 통계
    total_distance = 0
    velocities = []
    for i in range(1, len(positions_meters)):
        x1, y1 = positions_meters[i - 1]
        x2, y2 = positions_meters[i]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance
        velocities.append(distance)

    # 속도를 m/s 단위로 변환 (30fps 가정)
    velocities_ms = [v * 30 for v in velocities]

    print("\n=== 움직임 분석 결과 ===")
    print(f"총 이동 거리: {total_distance:.2f}m")
    print(f"평균 이동 속도: {np.mean(velocities_ms):.2f} m/s")
    print(f"최대 이동 속도: {np.max(velocities_ms):.2f} m/s")

    # 2. 코트 활용 분석
    left_half = len([x for x in x_coords_m if x < 4])  # 4m = 중앙선
    right_half = len([x for x in x_coords_m if x >= 4])
    front_half = len([y for y in y_coords_m if y < 4])
    back_half = len([y for y in y_coords_m if y >= 4])

    print("\n=== 코트 활용 분석 ===")
    print(f"좌측 점유율: {left_half/len(positions_meters)*100:.1f}%")
    print(f"우측 점유율: {right_half/len(positions_meters)*100:.1f}%")
    print(f"전방 점유율: {front_half/len(positions_meters)*100:.1f}%")
    print(f"후방 점유율: {back_half/len(positions_meters)*100:.1f}%")


def save_analysis_results(positions, scoring_events, filename=None):
    """분석 결과를 파일로 저장"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"movement_analysis_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        # 득점/실점 통계
        scores = len([e for e in scoring_events if e[0] == "득점"])
        concedes = len([e for e in scoring_events if e[0] == "실점"])

        f.write("=== 득점/실점 통계 ===\n")
        f.write(f"총 득점: {scores}\n")
        f.write(f"총 실점: {concedes}\n\n")

        # 득점/실점 상세 기록
        f.write("=== 득점/실점 상세 기록 ===\n")
        for i, (event_type, pos) in enumerate(scoring_events, 1):
            pos_meters = pixels_to_meters(pos, 640, 360)
            f.write(
                f"{i}. {event_type}: 위치 ({pos_meters[0]:.2f}m, {pos_meters[1]:.2f}m)\n"
            )

        f.write("\n")

        # 움직임 분석 결과도 파일에 추가
        valid_positions = [pos for pos in positions if pos != (-1, -1)]
        positions_meters = [pixels_to_meters(pos, 640, 360) for pos in valid_positions]

        total_distance = 0
        velocities = []
        for i in range(1, len(positions_meters)):
            x1, y1 = positions_meters[i - 1]
            x2, y2 = positions_meters[i]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance += distance
            velocities.append(distance)

        velocities_ms = [v * 30 for v in velocities]

        f.write("=== 움직임 분석 결과 ===\n")
        f.write(f"총 이동 거리: {total_distance:.2f}m\n")
        f.write(f"평균 이동 속도: {np.mean(velocities_ms):.2f} m/s\n")
        f.write(f"최대 이동 속도: {np.max(velocities_ms):.2f} m/s\n")

    print(f"\n분석 결과가 {filename} 파일로 저장되었습니다.")