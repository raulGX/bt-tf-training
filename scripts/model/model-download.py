import os.path
from six.moves import urllib
import sys
import tarfile

MOBILENETS_URL = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz'
MODEL_DOWNLOAD_PATH = '/Users/raulpopovici/Desktop/licenta/bt-tf-training/model/downloads'


def get_model_file(data_url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print('Successfully downloaded %s.', filename)
        print('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(output_dir)
    else:
        print('Model file present on disk')


def main():
    get_model_file(MOBILENETS_URL, MODEL_DOWNLOAD_PATH)


if __name__ == '__main__':
    main()
