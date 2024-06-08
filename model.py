import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model

model = load_model('model/realVsForged.h5')
class_names = ['Forged', 'Real']

def convert_to_ela_image(path, quality):
    original_image = Image.open(path).convert('RGB')
    resaved_file_name = 'resaved_image.jpg'
    original_image.save(resaved_file_name, 'JPEG', quality=quality)
    resaved_image = Image.open(resaved_file_name)

    ela_image = ImageChops.difference(original_image, resaved_image)
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference == 0:
        max_difference = 1
    scale = 350.0 / max_difference
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def prepare_image(image_path):
    image_size = (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def predict_image(image_path):
    image = prepare_image(image_path).reshape(-1, 128, 128, 3)
    prediction = model.predict(image)
    class_index = int(np.round(prediction)[0][0])
    confidence = (prediction[0][0] if class_index == 1 else 1 - prediction[0][0]) * 100
    return class_names[class_index], confidence
