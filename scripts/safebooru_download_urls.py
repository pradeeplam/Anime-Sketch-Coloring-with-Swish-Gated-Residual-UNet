#!/usr/bin/env python3

"""Downloads images from the Safebooru CSV located at:
kaggle.com/alamson/safebooru
"""
import argparse
import os
import subprocess
from multiprocessing import Pool


def download(urls_out):

    url, out_dirname = urls_out
    out_fname = os.path.join(out_dirname, url.split('/')[-1])

    if os.path.isfile(out_fname):
        return
    try:
        command = 'wget -q -o /dev/null -O {} {}'.format(out_fname, url)
        subprocess.check_call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass


def get_urls(csv_fname):
    urls = []
    first = True
    for line in open(csv_fname, 'r').readlines():
        # Skip header line without loading everything into memory
        if first:
            first = False
            continue
        url = line.split(',')[4].replace('"', '').strip().rstrip('\n')
        urls.append(url)
    return urls


def main(args):
    with Pool(4) as pool:
        urls = get_urls(args.csv_fname)
        # Ugly hack to pass multiple parameters to download()
        urls_out = [(url, args.out_dirname) for url in urls]
        pool.map(download, urls_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_fname', help='Safebooru CSV filename')
    parser.add_argument('out_dirname', help='Output directory name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
