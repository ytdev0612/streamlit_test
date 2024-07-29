import cv2

def draw(image, boxes, confidences, labels):
    print("\n최종 결정:")
    if len(boxes) > 0:
        for box, conf, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = map(int, box)
            label_name = label

            # 사각형 테두리 그리는 함수(사각형을 그릴 이미지, 사각형 좌상단 좌표, 사각형 우하단 좌표, 사각형 색, 사각형 두께)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 텍스트 추가하는 함수(텍스트 추가할 이미지, 텍스트 문자열, 텍스트 시작점, 텍스트 폰트, 텍스트 크기, 텍스트 색상, 텍스트 두께)
            cv2.putText(image, f'{label_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # 박스 좌표, 신뢰도, 라벨이름 출력
            print(f'박스 좌표: {box}, 신뢰도: {conf}, 라벨: {label_name}')