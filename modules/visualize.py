#! Projects/modules/visualize.py

import platform
import matplotlib.pyplot as plt
import numpy as np
from .calculates import pixels_to_meters


def draw_octagonal_court(plt):
    """8m x 8m 팔각형 경기장 그리기"""
    court_size = 8  # 코트 안쪽 크기 8m x 8m
    border_width = 0.6  # 주황색 부분 너비 60cm (0.6m)

    # 팔각형 꼭지점 계산
    x_outer = [court_size * np.cos(np.pi / 4 * i) for i in range(8)]
    y_outer = [court_size * np.sin(np.pi / 4 * i) for i in range(8)]

    # 주황색 외곽선 (0.6m 안쪽)
    inner_size = court_size - border_width  # 0.6m 안쪽으로 줄임
    x_inner = [inner_size * np.cos(np.pi / 4 * i) for i in range(8)]
    y_inner = [inner_size * np.sin(np.pi / 4 * i) for i in range(8)]

    # 경기장 외곽선 (검정색)
    plt.plot(x_outer + [x_outer[0]], y_outer + [y_outer[0]], "k-", linewidth=2)

    # 주황색 외곽선
    plt.plot(x_inner + [x_inner[0]], y_inner + [y_inner[0]], "orange", linewidth=2)

    # 중앙선
    plt.axhline(y=court_size / 2, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=court_size / 2, color="k", linestyle="--", alpha=0.3)

    # 경기 시작 위치 표시
    plt.plot(court_size / 2, court_size / 2, "ko", markersize=10)
    plt.plot(court_size / 2 - 1, court_size / 2, "ko", markersize=10)
    plt.plot(court_size / 2 + 1, court_size / 2, "ko", markersize=10)


def visualize_movement_pattern(
    positions, scoring_events, court_width=8, court_height=8
):
    # 한글 입력을 위한 플랫폼 확인
    system_check = platform.system()
    
    if system_check == "Darwin" or system_check == "Linux":
        plt.rcParams["font.family"] = "NanumGothic"
    elif system_check == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    
    # 한글 폰트 사용 시, 마이너스 기호 깨짐 방지    
    plt.rcParams["axes.unicode_minus"] = False

    if not positions:
        print("분석할 위치 데이터가 없습니다.")
        return

    # 트래킹 중단 지점(-1, -1)을 기준으로 이동 경로 분리
    segments = []
    current_segment = []

    for pos in positions:
        if pos == (-1, -1):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(pos)

    if current_segment:
        segments.append(current_segment)

    plt.figure(figsize=(12, 5))

    # 1. 이동 경로 (미터 단위)
    plt.subplot(1, 2, 1)

    # 팔각형 경기장 그리기
    draw_octagonal_court(plt)

    # 전체 이동 경로를 하나로 표시
    all_valid_positions = []
    for segment in segments:
        segment_meters = [
            pixels_to_meters(pos, court_width * 100, court_height * 100)
            for pos in segment
        ]  # 픽셀을 미터로 변환
        x_coords_m, y_coords_m = zip(*segment_meters)
        plt.plot(x_coords_m, y_coords_m, "-", color="blue", alpha=0.5)
        all_valid_positions.extend(segment_meters)

    # 전체 경로의 시작점과 끝점만 표시
    if all_valid_positions:
        plt.plot(
            all_valid_positions[0][0],
            all_valid_positions[0][1],
            "go",
            markersize=8,
            label="시작",
        )
        plt.plot(
            all_valid_positions[-1][0],
            all_valid_positions[-1][1],
            "ro",
            markersize=8,
            label="종료",
        )

    # 득점/실점 위치 표시
    for event_type, pos in scoring_events:
        pos_meters = pixels_to_meters(pos, court_width * 100, court_height * 100)
        color = "g" if event_type == "득점" else "r"
        marker = "+" if event_type == "득점" else "x"
        plt.plot(
            pos_meters[0],
            pos_meters[1],
            marker,
            color=color,
            markersize=12,
            markeredgewidth=2,
            label=f"{event_type}",
        )

    plt.title("선수 이동 경로 및 득점/실점 위치")
    plt.xlabel("거리 (m)")
    plt.ylabel("거리 (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # 정사각형 비율 유지

    # 2. 히트맵 (미터 단위)
    plt.subplot(1, 2, 2)
    valid_positions = [pos for pos in positions if pos != (-1, -1)]
    positions_meters = [
        pixels_to_meters(pos, court_width * 100, court_height * 100)
        for pos in valid_positions
    ]
    x_coords_m, y_coords_m = zip(*positions_meters)

    heatmap, xedges, yedges = np.histogram2d(
        x_coords_m,
        y_coords_m,
        bins=[20, 20],
        range=[[0, court_width], [0, court_height]],
    )

    plt.imshow(
        heatmap.T,
        origin="lower",
        extent=[0, court_width, 0, court_height],
        cmap="YlOrRd",
    )
    plt.colorbar(label="위치 빈도")

    # 히트맵 위에 팔각형 경기장 그리기
    draw_octagonal_court(plt)

    plt.title("위치 히트맵")
    plt.xlabel("거리 (m)")
    plt.ylabel("거리 (m)")
    plt.axis("equal")  # 정사각형 비율 유지

    plt.tight_layout()
    plt.show()

