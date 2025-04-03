#! Projects/modules/track_player_movement.py

import cv2

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

                    # 이동 경로 및 방향 표시
                    if len(positions) > 1:
                        prev_x, prev_y = positions[-2]
                        dx = center_x - prev_x
                        dy = center_y - prev_y


                        # 🔴 이동 경로 선으로 표시 (히트맵용)
                        cv2.line(frame, (prev_x, prev_y), (center_x, center_y), (0, 0, 255), 1)
    
                        # 작은 흔들림 무시
                        if abs(dx) > 2 or abs(dy) > 2:
                            cv2.line(frame, (prev_x, prev_y), (center_x, center_y), (0, 0, 255), 1)

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
