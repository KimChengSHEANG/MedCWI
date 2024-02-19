from src.paths import RESOURCES_DIR

import urllib.request
from tqdm import tqdm
from pathlib import Path
import tempfile
import gzip
import shutil
import os

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    tmp_filepath = tempfile.mktemp()
    # print(tmp_filepath)
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=tmp_filepath, reporthook=t.update_to)
    shutil.move(tmp_filepath, output_path)

def download_fasttext():
    filename = 'cc.fr.300.vec'
    output_dir = RESOURCES_DIR / 'others'
    download_filepath = output_dir / Path(filename + '.gz')
    output_filepath = output_dir / filename
    if not output_filepath.exists():
        download_url('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz', download_filepath)
        print('Extracting...')
        # if not output_filepath.exists():
        with gzip.open(download_filepath, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(download_filepath)
        print('Downloading FastText is completed!')
        