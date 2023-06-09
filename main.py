import cv2
from PIL import Image
from pytesseract import pytesseract


def extract_text(path_to_tesseract, image_path):
    pytesseract.tesseract_cmd = path_to_tesseract
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


path_to_tesseract = r"C:\Users\clash\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
image_path = r"image.png"

# Load the image
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to black and white
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


text = extract_text(path_to_tesseract, image_path)
print(text[:-1])