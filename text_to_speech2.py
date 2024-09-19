import threading
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
        # print("Input shapes:", self.input_shapes)
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # print("Input names:", self.input_names)
        preds = self.model.run(None, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]

        return text


directory = "Models/03_handwriting_recognition/202308060824"
configs = BaseModelConfigs.load(f"{directory}/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)


def read_text(text_):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty("rate", 150)  # Speed of speech
    engine.setProperty("volume", 0.9)  # Volume level (0.0 to 1.0)

    # Speak the given text
    engine.say(text_)
    engine.runAndWait()


def recognize_text(image, scale_factor=2.0, color=(0, 255, 0), thickness=1, show_steps=False, show_image=False,
                   sub_x=10, sub_y=10, greyscale_threshold=128):
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

    def convert_to_two_color(image_array, threshold=greyscale_threshold):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        return binary_image

    def generate_rectangles(image_, width, height):
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

    def target_wait_key():
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def arrange_in_reading_order(rectangles):
        """
        Arranges rectangles in reading order (left-to-right, top-to-bottom), grouping them by intersecting y-ranges.

        :param rectangles: List of rectangles where each rectangle is represented as
                           (x, y, width, height), with (x, y) as the top-left corner.
        :return: A list of rectangles sorted in reading order.
        """

        def y_ranges_intersect(rect1, rect2):
            """
            Check if the y-range of rect1 intersects with the y-range of rect2.
            """
            _, y1, _, h1 = rect1[:4]
            _, y2, _, h2 = rect2[:4]
            # If the y-ranges overlap, there is an intersection
            return max(y1, y2) <= min(y1 + h1, y2 + h2)

        # Sort the rectangles by their x and y coordinates to roughly arrange them
        rectangles.sort(key=lambda r: (r[1], r[0]))  # First sort by y, then by x

        # Grouping rectangles by intersecting y-ranges
        sorted_rectangles = []
        current_row = []

        for rect in rectangles:
            if not current_row:
                current_row.append(rect)
            else:
                # Check if the y-range of the current rectangle intersects with the last one in the current row
                if y_ranges_intersect(current_row[-1], rect):
                    current_row.append(rect)
                else:
                    # Sort the current row by x-coordinate before adding it to the final list
                    current_row = sorted(current_row, key=lambda r: r[0])  # Sort by x (left-to-right)
                    sorted_rectangles.extend(current_row)
                    current_row = [rect]  # Start a new row

        # Sort and append the last row
        if current_row:
            current_row = sorted(current_row, key=lambda r: r[0])  # Sort by x (left-to-right)
            sorted_rectangles.extend(current_row)

        return sorted_rectangles

    scaled_image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    original_image = scaled_image.copy()
    two_color_image = convert_to_two_color(scaled_image)
    bounding_boxes = generate_rectangles(two_color_image, width=sub_x, height=sub_y)
    labelled_bounding_boxes = check_and_update_bounding_rectangles(two_color_image, bounding_boxes)
    merged_and_labelled = merge_close_rectangles(labelled_bounding_boxes, threshold=1)
    arranged_in_reading_order = arrange_in_reading_order(merged_and_labelled)

    if show_steps:
        cv2.imshow("Original", original_image)
        cv2.imshow("Processed", two_color_image)
        cv2.imshow("Rectangles", draw_bounding_boxes(two_color_image, rectangles=labelled_bounding_boxes))
        cv2.imshow("Final", draw_bounding_boxes(two_color_image, rectangles=merged_and_labelled))
    if show_image:
        cv2.imshow("Bounded text", draw_bounding_boxes(original_image, rectangles=merged_and_labelled))

    detected_text = ""
    # for img in get_sub_images(original_image, arranged_in_reading_order):
    for img in get_sub_images(cv2.cvtColor(two_color_image, cv2.COLOR_GRAY2RGB), arranged_in_reading_order):
        prediction_text = model.predict(img)
        detected_text += prediction_text + " "
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_text


def capture_image(webcam_index=0):
    # Open the webcam
    cap = cv2.VideoCapture(webcam_index)

    if not cap.isOpened():
        print("Error: Could not open the webcam")
        return

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Display the frame in a window
        cv2.imshow(f"Webcam {webcam_index}", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # Capture an image when spacebar is pressed
        if key == ord(" "):
            cv2.imwrite("Test_images/image1.jpg", frame)
            cap.release()
            cv2.destroyAllWindows()
            return frame


def get_image(image_source="webcam"):
    if image_source.lower()[0:6] == "webcam":
        webcam_index = image_source[6:]
        if not webcam_index:
            webcam_index = 0
        else:
            webcam_index = int(webcam_index)
        return capture_image(webcam_index=webcam_index)
    else:
        return cv2.imread(image_source)


if __name__ == "__main__":
    # Set source to either "webcam" or a directory

    # Set source below. Use "webcam" for the webcam or image file path
    source = "Datasets/image4.jpg"
    # source = "webcam"

    image = get_image(image_source=source)

    if source == "webcam":
        # Set the "show_image" argument to True if you want to display the image being used
        # Set the "show_steps" argument to False if you would like to show the steps for the image processing
        # Increase the scale factor if words are too close together
        text = recognize_text(image=image, scale_factor=1.0,
                              show_steps=True, show_image=True, thickness=2, sub_x=10, sub_y=10,
                              greyscale_threshold=90)
        print("DETECTED:", text)
        read_text(f"{text}")

    else:
        # Set the "show_image" argument to True if you want to display the image being used
        # Set the "show_steps" argument to False if you would like to show the steps for the image processing
        # Increase the scale factor if words are too close together
        text = recognize_text(image=image, scale_factor=2.5,
                              show_steps=True, show_image=True, thickness=2, sub_x=10, sub_y=10)
        print("DETECTED:", text)
        read_text(f"{text}")
