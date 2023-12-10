import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import copy


def load_image():
    folder = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img


def save_image(image, name):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    folder = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(folder, name), image)


def get_labels():
    folder = os.path.dirname(os.path.abspath(__file__))
    folder += "/GoogleNet"
    labels = None
    prototxt = None
    model = None
    for file in os.listdir(folder):
        if "synset" in file:
            labels = open(os.path.join(folder, file))
        elif "prototxt" in file:
            prototxt = os.path.join(folder, file)
        elif "caffemodel" in file:
            model = os.path.join(folder, file)
    return labels, prototxt, model


def cut_images(image):
    x = 247
    y = 247
    start_point = (x, y)
    end_point = (x + 1440, y + 720)

    cropped_image = image[start_point[1] : end_point[1], start_point[0] : end_point[0]]

    return cropped_image


def detect(size):
    boxes_cats = []
    scores_cats = []
    boxes_dogs = []
    scores_dogs = []
    for i in range(0, cropped_image.shape[0], size):
        for j in range(0, cropped_image.shape[1], size):
            blob = cv2.dnn.blobFromImage(
                cropped_image[i : i + size, j : j + size],
                1,
                (224, 224),
                (104, 117, 123),
            )
            net.setInput(blob)
            preds = net.forward()
            idx = np.argsort(preds[0])[::-1][0]
            if (
                " cat" in classes[idx]
                or "Angora" in classes[idx]
                and preds[0][idx] > 0.9
            ):
                cv2.rectangle(
                    cropped_image,
                    (j, i + 5),
                    (j + size - 2, i + size - 5),
                    (255, 0, 0),
                    2,
                )
                boxes_cats.append((j, i, j + size, i + size))
                scores_cats.append(preds[0][idx])
                text = "CAT"
                cv2.putText(
                    cropped_image,
                    text,
                    (j + 10, i + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
            elif (
                " dog" in classes[idx]
                or "Samoyed" in classes[idx]
                and preds[0][idx] > 0.9
            ):
                cv2.rectangle(
                    cropped_image,
                    (j + 5, i + 5),
                    (j + size - 5, i + size - 5),
                    (255, 255, 0),
                    2,
                )
                boxes_dogs.append((j, i, j + size, i + size))
                scores_dogs.append(preds[0][idx])
                text = "DOG"
                cv2.putText(
                    cropped_image,
                    text,
                    (j + 10, i + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
    return boxes_cats, scores_cats, boxes_dogs, scores_dogs


def apply_non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.8, iou_threshold)
    filtered_boxes = []
    for i in indices:
        filtered_boxes.append(boxes[i])
    return filtered_boxes


if __name__ == "__main__":
    image = load_image()
    cropped_image = cut_images(image)
    copy_image = copy.deepcopy(cropped_image)
    plt.imshow(cropped_image)
    plt.show()

    labels, prototxt, model = get_labels()
    rows = labels.read().strip().split("\n")
    classes = [r[r.find(" ") + 1 :].split(",")[0] for r in rows]
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    boxes_cats1, scores_cats1, boxes_dogs1, scores_dogs1 = detect(180)
    boxes_cats2, scores_cats2, boxes_dogs2, scores_dogs2 = detect(360)
    boxes_cats3, scores_cats3, boxes_dogs3, scores_dogs3 = detect(720)

    boxes_cats = boxes_cats1 + boxes_cats2 + boxes_cats3
    scores_cats = scores_cats1 + scores_cats2 + scores_cats3
    boxes_dogs = boxes_dogs1 + boxes_dogs2 + boxes_dogs3
    scores_dogs = scores_dogs1 + scores_dogs2 + scores_dogs3

    plt.imshow(cropped_image)
    plt.show()
    save_image(cropped_image, "output.jpg")

    filtered_boxes = apply_non_max_suppression(boxes_cats, scores_cats)
    filtered_boxes += apply_non_max_suppression(boxes_dogs, scores_dogs)
    for box in filtered_boxes:
        cv2.rectangle(copy_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    plt.imshow(copy_image)
    plt.show()
