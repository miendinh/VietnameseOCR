#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import random
import sys
from config import *


class DataGenerator:
    def __init__(self):
        self.i = 0
        self.log = []
        self.errors = []
        self.data_folder = DATASET_DIR
        self.font_list = FONT_LIST
        self.data_set_csv = DATASET_FILE
        self.characters = []
        self.dataset_size = 0

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def get_list_characters(self):
        if len(self.characters) != 0:
            return self.characters
        else:
            characters = []
            with open(CHARACTERS_SET) as cf:
                for r in cf:
                    if ',,' in r:
                        c = ','
                    else:
                        _, c = r.split(',')
                    characters.append(c.replace('\n', ''))

            self.characters = characters
            return characters

    def create_text_image(self, text, font_ttf, idx_category, font_size):
        try:
            image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_ttf, font_size)
            w, h = draw.textsize(text, font=font)
            draw.text(((IMG_WIDTH - w) / 2, (IMG_HEIGHT - h) / 2), text, (0, 0, 0), font=font)

            if SAVE_TEXT_IMAGE_TO_DISK:
                image.save(self.data_folder + str(idx_category) + '/' + str(self.i) + '.jpg')

            self.log.append({'font': font_ttf, 'image': str(self.i) + '.jpg'})
            self.i = self.i + 1
            return image
        except Exception as e:
            self.errors.append({'font': font_ttf, 'errors': str(e)})
            return None

    def generate_data_set(self, text, idx_category):
        images = []
        with open(self.font_list, 'r') as fonts:
            for font in fonts:
                if '#' not in font:
                    for font_size in range(FONT_SIZE_MIN, FONT_SIZE_MAX + 1):
                        image = self.create_text_image(text, font.replace('\n', ''), idx_category, font_size)
                        if image != None:
                            self.dataset_size = self.dataset_size + 1
                            images.append(image)
        self.i = 0
        return images

    def generate_dataset(self):
        characters = self.get_list_characters()
        for idx, ch in enumerate(characters):
            if SAVE_TEXT_IMAGE_TO_DISK and not os.path.exists(self.data_folder + str(idx)):
                os.makedirs(self.data_folder + str(idx))

            c_images = self.generate_data_set(ch, idx)
            print('.', end='')
            for ic in c_images:
                image = np.asarray(ic)
                image = self.rgb2gray(image)
                image = image.reshape(1, IMG_WIDTH * IMG_HEIGHT)
                with open(DATASET_FILE, 'ab') as df:
                    image = np.concatenate((image, np.array([[int(idx)]])), axis=1)
                    np.savetxt(df, image, delimiter=",", fmt="%d")


if __name__ == "__main__":
    print('Generating dataset...')
    generator = DataGenerator()
    generator.generate_dataset()
    print('Text Image Dataset is generated:', DATASET_FILE_PATH)
