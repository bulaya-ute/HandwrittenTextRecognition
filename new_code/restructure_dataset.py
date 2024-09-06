import os
import shutil
from tqdm import tqdm
from PIL import Image

# Define paths
words_dir = "../Datasets/IAM_Words/words"
words_txt_path = os.path.join(words_dir, "words.txt")
output_dir = f"{os.path.dirname(words_dir)}/Datasets/IAM_Words_restructured"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parse the words.txt file and organize images
with open(words_txt_path, "r") as file:
    for line in tqdm(file.readlines(), desc="Restructuring dataset"):
        if line.startswith("#"):
            continue  # Skip comment lines

        parts = line.strip().split(" ")
        if len(parts) < 9:
            continue

        filename = parts[0]  # Get image filename (without extension)
        x, y, w, h = map(int, parts[3:7])  # Get bounding box coordinates
        word = parts[-1]  # Get the word label

        if any([c in word for c in ("\"", "'", ":", "?", "*")]):
            continue

        # Construct paths
        subdir = filename.split("-")[0]  # e.g., a01
        img_filename = filename + ".png"
        img_src_path = os.path.join(words_dir, subdir, filename.split("-")[1], img_filename)
        word_dir = os.path.join(output_dir, word)
        img_dest_path = os.path.join(word_dir, img_filename)

        # Create the word directory if it doesn't exist
        os.makedirs(word_dir, exist_ok=True)

        # Crop and save the image to the corresponding word directory
        if os.path.exists(img_src_path):
            with Image.open(img_src_path) as img:
                cropped_img = img.crop((x, y, x + w, y + h))  # Crop using bounding box
                cropped_img.save(img_dest_path)

print("Dataset restructured and cropped successfully!")
