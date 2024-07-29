from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from torchvision.ops import nms
import torch

# 모든 모델 예측 수행, 결과 담기
def ensemble_predict(image):
    results = []
    detection_counts = []
    for model in models:
        print(f'\n{model[0]}', end="")
        result = model[1].predict(image, conf=0.5)
        results.append(result)
        # 각 모델에서 감지한 바운딩 박스의 수
        detection_counts.append(len(result[0].boxes))
    print('\n각 모델별 감지한 객체 수:')
    print(detection_counts)

    if sum(detection_counts) == 0:
        print("모든 모델이 객체를 감지하지 못함.")
        return [], [], []

    # 과반수 이상의 모델이 바운딩 박스가 없는 경우
    if sum(count == 0 for count in detection_counts) >= len(models) / 2: # count가 0인 모델의 수가 과반수 이상인지 검사
        print("\n과반수 이상의 모델이 객체를 감지하지 못하여 화면에 표시하지 않음.")
        return [], [], []

    combined_results = combine_results(*results)  # final_boxes, final_confidences, final_labels
    return combined_results

# 각 결과에서 바운딩 박스, 신뢰도, 라벨 추출
def combine_results(*results):
    combined_boxes = []
    combined_confidences = []
    combined_labels = defaultdict(list)  # 같은 박스 위치에 여러 레이블을 저장하기 위한 딕셔너리

    # 각 모델의 결과에서 바운딩 박스를 추출하여 결합
    for model_index, result_list in enumerate(results):
        model_name = models[model_index][0]
        for result in result_list:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                # 클래스 번호를 클래스 이름으로 변환
                class_name = model_names[model_name][class_id]

                combined_boxes.append([x1, y1, x2, y2])
                combined_confidences.append(conf)
                combined_labels[(x1, y1, x2, y2)].append(class_name)

    combined_boxes = np.array(combined_boxes)
    combined_confidences = np.array(combined_confidences)

    # 박스를 그룹화하여 다수결로 라벨 결정
    final_boxes = []
    final_confidences = []
    final_labels = []

    box_groups = group_boxes_by_overlap(combined_boxes)  # 겹치는 박스 번호들끼리 그룹화한 리스트

    if len(box_groups) > 0:
        for group in box_groups:
            group_boxes = combined_boxes[group]  # 그룹의 각 번호에 해당하는 박스들의 좌표배열
            group_confidences = combined_confidences[group]  # 그룹의 각 번호에 해당하는 박스들의 신뢰도
            group_labels = [combined_labels[tuple(box)] for box in group_boxes]  # combined_labels = {(박스 좌표배열) : 라벨이름}

            # 그룹 내에서 평균 박스와 평균 신뢰도를 계산
            avg_box = np.mean(group_boxes, axis=0)
            avg_conf = np.mean(group_confidences)

            # 각 박스에 대해 다수결로 레이블 결정
            flattened_labels = [label for sublist in group_labels for label in sublist]  # 라벨이름들이 저장된 리스트
            flattened_labels = [label.lower() for label in flattened_labels] # 라벨이름 모두 소문자로 통일
            print(flattened_labels)
            labels_counter = Counter(flattened_labels) # 각 라벨의 빈도수 조사
            max_count = max(labels_counter.values()) # 최빈값
            most_common_label = [item for item, count in labels_counter.items() if count == max_count] # 최빈값에 해당하는 라벨
            # 최빈값이 1보다 크고, 빈도수가 가장 큰 라벨이 한개인 경우
            if max_count > 1 and len(most_common_label) == 1:
                final_boxes.append(avg_box)
                final_confidences.append(avg_conf)
                final_labels.append(most_common_label[0])
            # 최빈값이 같은 라벨이 여러 개인 경우, 그 라벨들의 박스끼리만 NMS 적용
            elif max_count > 1 and len(most_common_label) > 1:
                label_indices = []
                # 라벨로 박스번호 필터링
                for label in most_common_label:
                    for i, l in enumerate(flattened_labels):
                        if l == label:
                            label_indices.append(i)
                    
                label_boxes = group_boxes[label_indices] # 최빈값이 같은 박스들
                label_confidences = group_confidences[label_indices] # 최빈값이 같은 박스들의 신뢰도
                
                # NMS 적용
                indices = nms(torch.tensor(label_boxes), torch.tensor(label_confidences), 0.4)
                final_boxes.append(label_boxes[indices])
                final_confidences.append(label_confidences[indices])
                final_labels.append(flattened_labels[indices])
            # 라벨의 빈도수가 모두 1일 경우
            else: # 모두 NMS 적용
                indices = nms(torch.tensor(group_boxes), torch.tensor(group_confidences), 0.4)
                final_boxes.append(group_boxes[indices])
                final_confidences.append(group_confidences[indices])
                final_labels.append(flattened_labels[indices])

        final_boxes = np.array(final_boxes)
        final_confidences = np.array(final_confidences)
    else:
        return [], [], []

    return final_boxes, final_confidences, final_labels

# 박스가 겹치는 그룹을 찾는 함수(각 그룹 안에는 박스 번호들이 있음)
def group_boxes_by_overlap(boxes, iou_threshold=0.4):
    distances = cdist(boxes, boxes, lambda x, y: 1 - iou(x, y))  # 두 박스간의 겹침 정도(비율)가 클수록 1에서 뺀 값이 작아지므로 거리가 작아진다.
    groups = []
    visited = set()

    for i in range(len(boxes)):  # i: 현재 박스의 인덱스
        if i in visited:  # 이미 방문된 박스 집합(visited)에 있다면 다음 반복으로 넘어감
            continue
        group = [i]  # 현재 박스 번호를 포함한 그룹 생성
        visited.add(i)  # 현재 박스 번호를 집합에 추가
        for j in range(i + 1, len(boxes)):  # 현재 박스의 다음 박스 번호부터 순회
            if j in visited:  # 이미 방문된 박스 집합(visited)에 있다면 다음 반복으로 넘어감
                continue
            if distances[i, j] < iou_threshold:  # 박스 i와 j의 거리가 iou_threshold보다 작다면, 두 박스가 많이 겹친다고 판단함
                group.append(j)  # 박스 번호 j를 group에 추가
                visited.add(j)  # 집합에도 추가
        groups.append(group)  # 그룹을 groups에 추가
    print('\n겹치는 박스끼리 박스 번호로 그룹화:')
    print(groups)

    groups = [group for group in groups if len(group) >= 3] # 그룹내 박스가 3개 이상인 것만 통과
    print('\n박스가 3개 이상인 그룹의 라벨:')
    return groups

# IoU (Intersection over Union) 계산 함수
# 객체 탐지 및 이미지 분석에서 두 바운딩 박스 간의 겹침 정도를 측정하는 지표,
# IoU는 두 박스의 교차 영역과 두 박스의 합집합 영역 간의 비율을 나타냄
def iou(box1, box2):
    x1 = max(box1[0], box2[0])  # x1 vs X1, 더 큰 값이 교집합 영역 좌상단 x좌표
    y1 = max(box1[1], box2[1])  # y1 vs Y1, 더 큰 값이 교집합 영역 좌상단 y좌표
    x2 = min(box1[2], box2[2])  # x2 vs X2, 더 작은 값이 교집합 영역 우하단 x좌표
    y2 = min(box1[3], box2[3])  # y2 vs Y2, 더 작은 값이 교집합 영역 우하단 y좌표

    intersection = max(0, x2 - x1) * max(0, y2 - y1)  # 교집합 영역 가로길이 * 교집합 영역 세로길이 = 교집합 영역의 전체 면적
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  # 박스1의 가로길이 * 세로길이 = 박스1의 전체 면적
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  # 박스2의 가로길이 * 세로길이 = 박스2의 전체 면적
    union = box1_area + box2_area - intersection  # 두 박스의 합집합 영역에서 교집합 영역을 뺀 값

    return intersection / union if union > 0 else 0  # 교집합 면적을 합집합 면적으로 나눈값


# 모델 파일명 리스트
model_files = [f'models/model{model_number}.pt' for model_number in [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]] # 밴: model8, model10

# 모델 로드 및 names 저장
models = []
model_names = {}

# 모델파일 이름, 모델, 클래스 저장
for model_file in model_files:
    model = YOLO(model_file)
    models.append((model_file, model)) # 모델 파일 이름도 같이 저장함
    model_names[model_file] = model.names