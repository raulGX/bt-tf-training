from pathlib import Path
import os
import scipy
from scipy import misc

###
# Contains a hashmap of image vectors
###


def load_data(image_dir):
    if not os.path.exists(image_dir):
        print('Image directory does not exist')
        return None

    result = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    for sub_dir in sub_dirs:
        if sub_dir != image_dir:
            classname = sub_dir.split('/')[-1]
            image_list = []
            for f in os.listdir(sub_dir):
                file_path = os.path.join(sub_dir, f)
                if os.path.isfile(file_path):
                    if 'DS_Store' in file_path:
                        continue
                    image = scipy.misc.imread(file_path)
                    image = image.astype(float)
                    image_list.append(image)

            result[classname] = image_list

    return result


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(
        dir_path, '..', 'download-dataset/downloads'))
    print(load_data(dir_path))


if __name__ == '__main__':
    main()
