import typing
import cv2
import numpy as np
import pyttsx3
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs


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


def read_text(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty("rate", 150)  # Speed of speech
    engine.setProperty("volume", 0.9)  # Volume level (0.0 to 1.0)

    # Speak the given text
    engine.say(text)
    engine.runAndWait()


def fill_bounding_rectangles(image, rectangles):
    result_image = np.ones_like(image) * 255  # Create a white image of the same size as input

    for rect in rectangles:
        x, y, w, h, _ = rect
        result_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]

    return result_image


def group_rectangles(rectangles):
    grouped_rectangles = []

    while rectangles:
        current_rect = rectangles.pop(0)
        group = [current_rect]

        i = 0
        while i < len(rectangles):
            x1, y1, w1, h1 = current_rect[:4]
            x2, y2, w2, h2 = rectangles[i][:4]
            if (
                abs(x2 - (x1 + w1)) <= 1
                or abs(x1 - (x2 + w2)) <= 1
                or abs(y2 - (y1 + h1)) <= 1
                or abs(y1 - (y2 + h2)) <= 1
            ):
                group.append(rectangles.pop(i))
            else:
                i += 1

        merged_rect = group[0]
        for rect in group[1:]:
            x1, y1, w1, h1 = merged_rect[:4]
            x2, y2, w2, h2 = rect[:4]
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1 + w1, x2 + w2)
            max_y = max(y1 + h1, y2 + h2)
            merged_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

        grouped_rectangles.append(group)

    return grouped_rectangles


def recognize_text(image, scale_factor=2.0, color=(0, 0, 255), thickness=1, show_steps=False, show_image=False):
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

    def is_rectangles_close(rect1, rect2, threshold=1):
        """Return True if two rects are 'threshold' units or less apart,
        else Return False"""
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

    def draw_bounding_boxes(image_, rectangles, color_=color, thickness_=thickness):
        image_with_boxes = image_.copy()

        for rect in rectangles:
            x, y, w, h, _ = rect
            cv2.rectangle(image_with_boxes, pt1=(x, y), pt2=(x + w, y + h), color=color_, thickness=thickness_)

        return image_with_boxes

    def get_sub_images(image_, bounding_rectangles):
        sub_images = []

        for rect in bounding_rectangles:
            x_, y_, w_, h_, _ = rect
            sub_image = image_[y_:y_ + h_, x_:x_ + w_]
            sub_images.append(sub_image)

        return sub_images

    def convert_to_two_color(image_array, threshold=128):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        return binary_image

    def generate_rectangles(image_, width, height):
        """Return a bunch of rectangles of specified dimensions that cover the entire image"""
        rows, cols = image_.shape
        num_rows = rows // height
        num_cols = cols // width
        rectangles = []
        for x in range(num_cols):
            for y in range(num_rows):
                x_coord = x * width
                y_coord = y * height
                rect_tuple = (x_coord, y_coord, width, height, 0)
                rectangles.append(rect_tuple)
        return rectangles

    def check_and_update_bounding_rectangles(image_, bounding_rectangles):
        """
        Check each rectangular region of format (x, y, w, h, l) for the presence
        of a black pixel. If one is found set l = 1.
        Return a list of rectangles with asserted 'l'.
        :param image_:
        :param bounding_rectangles:
        :return:
        """
        updated_bounding_boxes = []
        for i, (x, y, w, h, l) in enumerate(bounding_rectangles):
            roi = image_[y:y + h, x:x + w]
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

    def sort_rectangles(rectangles):
        """Sort rectangles in reading order"""
        centers = []
        for i, rect in enumerate(rectangles):
            x, y, w, h = rect[:4]
            centers.append((x + w // 2, y + h // 2, i))

        sorted_centers = sorted(centers, key=lambda item: (item[0] + 1) ** (item[1] + 1))
        sorted_rects = [rectangles[sc[2]] for sc in sorted_centers]
        return sorted_rects

    scaled_image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    original_image = scaled_image.copy()
    two_color_image = convert_to_two_color(scaled_image)
    bounding_boxes = generate_rectangles(two_color_image, width=10, height=10)
    labelled_bounding_boxes = check_and_update_bounding_rectangles(two_color_image, bounding_boxes)
    labelled_bounding_boxes = sort_rectangles(labelled_bounding_boxes)
    merged_and_labelled = merge_close_rectangles(labelled_bounding_boxes, threshold=1)

    if show_steps:
        cv2.imshow("Original", original_image)
        cv2.imshow("Threshold", two_color_image)
        cv2.imshow("Text detection", draw_bounding_boxes(two_color_image, rectangles=labelled_bounding_boxes,
                                                         color_=(0, 0, 0), thickness_=1))
        cv2.imshow("Final", draw_bounding_boxes(two_color_image, rectangles=merged_and_labelled))
    if show_image:
        # for group in group_rectangles(rectangles=merged_and_labelled):
        #     cv2.imshow("Group", fill_bounding_rectangles(original_image, group))
        cv2.imshow("Bounded text", draw_bounding_boxes(original_image, rectangles=merged_and_labelled))

    detected_text = ""
    for img in get_sub_images(original_image, merged_and_labelled):
        prediction_text = model.predict(img)
        detected_text += prediction_text + " "
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_text


if __name__ == "__main__":
    directory = "Models/03_handwriting_recognition/202308060824"
    configs = BaseModelConfigs.load(f"{directory}/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    text = recognize_text(image=cv2.imread("Datasets/download.jpeg"), scale_factor=2.5,
                          show_steps=False, show_image=True, thickness=2, color=(255, 0, 0))
    print("DETECTED:", text)
    read_text(f"{text}")
