# Module to download the dataset.

import os

import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


if __name__ == '__main__':
    # The checkpoints can be downloaded at https://zenodo.org/record/3350871
    filenames = ["fma_rock_checkpoints.zip",
                 "Lakh_checkpoints.zip",
                 "maestro-long_checkpoints.zip",
                 "maestro-medium_checkpoints.zip",
                 "maestro-short_checkpoints.zip",
                 "midi_maestro_checkpoints.zip"
                ]

    url = 'https://zenodo.org/record/3897144/files/{}?download=1'


    md5s = ["529ed1593a6fc2315b12ade677d69e4b",
            "d28d85df194fe43b31fd7ccc7cea926d",
            "c2e8d85811bbf413b4c48c247d2e8541",
            "1ceb0c0fb9c86abdecd36324d33c4b7d",
            "1ac767255cf9edd53b8215a4301132dc",
            "61a1877a9b22784192173e6c6456ca6f"
            ]

    for filename, md5 in zip(filenames, md5s):

        print('Download checkpoints: {}'.format(filename))
        download(url.format(filename), './')
        assert(check_md5(filename, md5))
        print('Extract checkpoints: {}'.format(filename))
        unzip(filename, 'saved_results')
        os.remove(filename)