from google_images_download import google_images_download
import json
from PIL import Image
import os
import time
import tensorflow as tf


def verify_image(img_file, sess):
    # test image
    try:
        v_image = open(img_file, 'rb')
        return sess.run(tf.image.is_jpeg(v_image.read()))
        # is valid
        #print("valid file: "+img_file)
    except:
        return False


def read_config(file_name):
    with open(file_name) as f:
        data = json.load(f)
        return data


def main():
    config = read_config('config.json')
    defaultImageCount = config.get("defaultImageCount", 100)
    fetcher = google_images_download.googleimagesdownload()
    for label in config["labels"]:
        count = label.get("count", defaultImageCount)
        name = label['name']
        if 'suffix' in label:
            for suffix in label['suffix']:
                print(suffix)
                absolute_image_paths = fetcher.download(
                    {"keywords": name + ' ' + suffix, "image_directory": name, "limit": count, "format": "jpg"})

        else:
            absolute_image_paths = fetcher.download(
                {"keywords": name, "image_directory": name, "limit": count, "format": "jpg"})
    sess = tf.InteractiveSession()
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'downloads')):
        for file in files:
            currentFile = os.path.join(root, file)
            # print(currentFile)
            # test image
            if not verify_image(currentFile, sess):
                print('removed')
                os.remove(currentFile)


if __name__ == '__main__':
    main()
