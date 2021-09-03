import time
import datetime
from pathlib import Path

def main(pathname):
    p = Path(pathname)
    files = [x.name for x in p.iterdir() if x.suffix == '.JPG' or x.suffix == '.jpg']
    print(files)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the folder, where the images are located.')
    args = parser.parse_args()
    print(args.path)
    print('done!')
    series_id = 5
    for throw_id in range(5):
        print(f"magictest-{datetime.datetime.now():%Y%m%d-%H%M%S}-{series_id}-{throw_id}-image.JPG")
        time.sleep(1.1)
    # pathname = '/home/max/phd/radar/dart_detection'
    # pathname = '.'
    # main(pathname)

    # filename=f"magictest-{datetime.datetime.now():%Y%m%d-%H%M%S}-{series_id}-{throw_id}-position.yaml" # < use this file name
    # filename=f"magictest-{datetime.datetime.now():%Y%m%d-%H%M%S}-{series_id}-{throw_id}-image.yaml")