import typing

import cv2
import numpy as np
import pytesseract
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

# Set the path to the Tesseract executable (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]

        return text


def convert_to_two_color(image_array, threshold=128):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def draw_bounding_boxes(image, rectangles, color=(0, 255, 0), thickness=1):
    image_with_boxes = image.copy()

    for rect in rectangles:
        x, y, w, h, _ = rect
        cv2.rectangle(image_with_boxes, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)

    return image_with_boxes


def merge_rectangles(rect1, rect2):
    x1, y1, w1, h1, _ = rect1
    x2, y2, w2, h2, _ = rect2

    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)

    merged_width = max_x - min_x
    merged_height = max_y - min_y

    merged_rectangle = (min_x, min_y, merged_width, merged_height, 0)
    return merged_rectangle


def is_rectangles_close(rect1, rect2, threshold=5):
    x1, y1, w1, h1, _ = rect1
    x2, y2, w2, h2, _ = rect2

    x_min, x_max = x1 - threshold, x1 + w1 + threshold
    y_min, y_max = y1 - threshold, y1 + h1 + threshold

    corners = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]
    for x, y in corners:
        if x_min <= x <= x_max:
            if y_min <= y <= y_max:
                return True
    return False


def check_and_update_bounding_rectangles(image, bounding_rectangles):
    updated_bounding_boxes = []
    for i, (x, y, w, h, l) in enumerate(bounding_rectangles):
        roi = image[y:y + h, x:x + w]
        if np.any(roi == 0):
            bounding_rectangles[i] = (x, y, w, h, 1)
            updated_bounding_boxes.append((x, y, w, h, 1))

    return updated_bounding_boxes


def merge_close_rectangles(bounding_rectangles: list, threshold=20):
    merged_rectangles = bounding_rectangles.copy()
    changes_made = True
    if not merged_rectangles:
        return merged_rectangles

    while changes_made:
        # cv2.imshow("Current", draw_bounding_boxes(image_object, merged_rectangles))
        # cv2.waitKey(0)
        for i, box1 in enumerate(merged_rectangles):
            for j, box2 in enumerate(merged_rectangles):
                if i != j:
                    if is_rectangles_close(box1, box2, threshold):
                        merged_rectangle = merge_rectangles(box1, box2)
                        merged_rectangles[i] = merged_rectangle
                        del merged_rectangles[j]
                        # changes_made = True
                        break
            else:
                # changes_made = False
                continue
            break
        else:
            break
        continue
    return merged_rectangles


def generate_rectangles(image, width, height):
    rows, cols = image.shape

    num_rows = rows // height
    num_cols = cols // width

    rectangles = []

    for y in range(num_rows):
        for x in range(num_cols):
            x_coord = x * width
            y_coord = y * height
            rect_tuple = (x_coord, y_coord, width, height, 0)
            rectangles.append(rect_tuple)

    return rectangles


def grayscale_to_rgb(grayscale_image):
    # Get the height and width of the grayscale image
    height, width = grayscale_image.shape

    # Create an empty RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign grayscale values to all three channels
    rgb_image[:, :, 0] = grayscale_image
    rgb_image[:, :, 1] = grayscale_image
    rgb_image[:, :, 2] = grayscale_image

    return rgb_image


def get_sub_images(image, bounding_rectangles):
    sub_images = []

    for rect in bounding_rectangles:
        x_, y_, w_, h_, _ = rect
        sub_image = image[y_:y_ + h_, x_:x_ + w_]
        sub_images.append(sub_image)

    return sub_images


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    directory = "Models/03_handwriting_recognition/202308060824"
    configs = BaseModelConfigs.load(f"{directory}/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Specify the path to the image you want to process
    image_path = 'Datasets/image4.jpg'
    image_object = cv2.imread(image_path)

    # resize
    scale_factor = 2
    image_object = cv2.resize(image_object,
                              (image_object.shape[1] * scale_factor, image_object.shape[0] * scale_factor))

    cv2.imshow("Original", image_object)
    original_image = image_object.copy()

    image_object = convert_to_two_color(image_object)
    bounding_boxes = generate_rectangles(image_object, width=10, height=10)
    labelled_bounding_boxes = check_and_update_bounding_rectangles(image_object, bounding_boxes)
    merged_and_labelled = merge_close_rectangles(labelled_bounding_boxes, threshold=1)

    # cv2.imshow("Processed", image_object)
    # cv2.imshow("Rectangles", draw_bounding_boxes(image_object, rectangles=labelled_bounding_boxes))
    # cv2.imshow("Final", draw_bounding_boxes(image_object, rectangles=merged_and_labelled))
    cv2.imshow("Bounded text", draw_bounding_boxes(original_image, rectangles=merged_and_labelled, thickness=2))

    detected_text = ""
    for img in get_sub_images(original_image, merged_and_labelled):
        prediction_text = model.predict(img)
        detected_text += prediction_text + " "
    print(f"DETECTED TEXT:\n{detected_text}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
