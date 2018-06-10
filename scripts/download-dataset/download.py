from google_images_download import google_images_download
import json


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


if __name__ == '__main__':
    main()
