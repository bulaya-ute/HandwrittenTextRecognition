import cv2
import pytesseract
from pytesseract import Output

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

def extract_words_from_image(img):
    """
    Extracts words from the image, arranges them in reading order, and returns a list of cropped word images.

    :param img: Input image (as a NumPy array)
    :return: List of cropped word images arranged in reading order.
    """
    # Perform OCR and get bounding box details
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Number of detected text regions
    num_boxes = len(data['level'])

    # Extract bounding boxes and words at the word level (level 5)
    words = []
    rectangles = []
    for i in range(num_boxes):
        if data['level'][i] == 5:  # Level 5 corresponds to individual words
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            word = data['text'][i].strip()  # The detected word
            if word:  # Avoid empty or non-valid words
                words.append((word, x, y, w, h))
                rectangles.append((x, y, w, h))

    # Arrange the bounding boxes in reading order
    sorted_rectangles = arrange_in_reading_order(rectangles)

    # Create a list of cropped images (each corresponding to a word)
    cropped_images = []
    for (x, y, w, h) in sorted_rectangles:
        # Crop the word image using OpenCV
        cropped_img = img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)

    return cropped_images

# Example usage:
if __name__ == "__main__":
    # Load the image using OpenCV
    img = cv2.imread('Datasets/test_image.jpg')

    # Extract cropped word images in reading order
    cropped_word_images = extract_words_from_image(img)

    # Display each cropped word image for verification
    for i, cropped_img in enumerate(cropped_word_images):
        cv2.imshow(f'Word {i+1}', cropped_img)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
