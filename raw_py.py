import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def track_player_movement(video_path, court_width=640, court_height=360):
    cap = cv2.VideoCapture(video_path)
    positions = []
    scoring_events = []
    frame_count = 0
    fullscreen = False
    paused = False  # 일시정지 상태 추적

    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / fps)  # 프레임 간 기본 대기 시간 (밀리초)

    tracker = cv2.TrackerCSRT_create()

    ret, frame = cap.read()
    if not ret:
        print("비디오를 읽을 수 없습니다.")
        return [], []

    frame = cv2.resize(frame, (court_width, court_height))

    print("\n=== 트래킹 가이드 ===")
    print("1. 첫 프레임에서 트래킹할 선수를 마우스로 선택하세요.")
    print("2. 선수의 전신이 포함되도록 박스를 그려주세요.")
    print("3. 선택 후 SPACE 또는 ENTER 키를 눌러주세요.")
    print("4. 트래킹 중 조작법:")
    print("   - 'q': 종료")
    print("   - 'r': 선수 재선택")
    print("   - '1': 득점 표시")
    print("   - '2': 실점 표시")
    print("   - 'f': 전체화면 전환")
    print("   - 'space': 일시정지/재생")
    print("   - 's': 느리게 재생")
    print("   - 'd': 빠르게 재생")
    print("   - 'right arrow': 앞으로 이동 (30프레임)")

    window_name = "선수 트래킹"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, court_width, court_height)

    bbox = cv2.selectROI("트래킹할 선수 선택", frame, False)
    tracker.init(frame, bbox)

    current_speed = 1  # 재생 속도 (1 = 정상)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (court_width, court_height))
                frame_count += 1

                success, bbox = tracker.update(frame)

                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    center_x = x + w // 2
                    center_y = y + h // 2
                    positions.append((center_x, center_y))

                    # 이동 경로 표시
                    if len(positions) > 1:
                        for i in range(1, len(positions)):
                            if positions[i - 1] == (-1, -1):
                                continue
                            cv2.line(
                                frame, positions[i - 1], positions[i], (0, 0, 255), 1
                            )
                else:
                    # 트래킹 실패 시 재선택 안내
                    cv2.putText(
                        frame,
                        "트래킹 실패! 'r'키를 눌러 재선택하세요",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            # 상태 표시
            status = "일시정지" if paused else f"재생 중 (x{current_speed})"
            cv2.putText(
                frame,
                f"상태: {status}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # 득점/실점 이벤트 표시
            for event_type, event_pos in scoring_events:
                color = (0, 255, 0) if event_type == "득점" else (0, 0, 255)
                text = "+" if event_type == "득점" else "-"
                cv2.circle(frame, event_pos, 5, color, -1)
                cv2.putText(
                    frame,
                    text,
                    (event_pos[0] - 5, event_pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # 진행 상황 표시
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = (frame_count / total_frames) * 100
            cv2.putText(
                frame,
                f"진행률: {progress:.1f}%",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow(window_name, frame)

            # 키 입력 처리
            wait_time = int(delay / current_speed) if not paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                positions.append((-1, -1))
                bbox = cv2.selectROI("트래킹할 선수 선택", frame, False)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
            elif key == ord("1"):
                scoring_events.append(("득점", (center_x, center_y)))
                print(f"득점 기록: 위치 ({center_x}, {center_y})")
            elif key == ord("2"):
                scoring_events.append(("실점", (center_x, center_y)))
                print(f"실점 기록: 위치 ({center_x}, {center_y})")
            elif key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(
                        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                else:
                    cv2.setWindowProperty(
                        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(window_name, court_width, court_height)
            elif key == ord(" "):  # 스페이스바: 일시정지/재생
                paused = not paused
            elif key == ord("s"):  # 느리게
                current_speed = max(0.25, current_speed / 2)
            elif key == ord("d"):  # 빠르게
                current_speed = min(4, current_speed * 2)
            elif key == 83 or key == 100:  # 오른쪽 화살표 또는 'd' 키
                if paused:  # 일시정지 상태에서만 프레임 이동
                    frame_count += 30  # 30프레임 앞으로
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return positions, scoring_events


def pixels_to_meters(
    pixel_coords,
    court_width_pixels,
    court_height_pixels,
    court_width_meters=8,
    court_height_meters=8,
):
    """픽셀 좌표를 실제 미터 단위로 변환"""
    x_pixel, y_pixel = pixel_coords
    # 중앙을 기준으로 변환
    x_meters = (x_pixel - (court_width_pixels / 2)) * (
        court_width_meters / court_width_pixels
    )
    y_meters = ((court_height_pixels / 2) - y_pixel) * (
        court_height_meters / court_height_pixels
    )
    return (x_meters, y_meters)


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


def main():
    # 비디오 파일 경로 설정
    video_dir = r"C:\Users\USER\OneDrive\바탕 화면\16. 전력분석관\1. 2024\1. 대회영상\2. 편집영상\1. 2024_paralympic\1. male\-58kg"
    video_name = "parataekwondo_paralymic_ESP_TUR(8강).mp4"
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

        # 득점/실점 통계 출력
        print("\n=== 득점/실점 통계 ===")
        scores = len([e for e in scoring_events if e[0] == "득점"])
        concedes = len([e for e in scoring_events if e[0] == "실점"])
        print(f"총 득점: {scores}")
        print(f"총 실점: {concedes}")

        # 분석 결과 저장
        save_analysis_results(positions, scoring_events)


if __name__ == "__main__":
    main()
