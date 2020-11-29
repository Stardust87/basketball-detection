import os
from pathlib import Path

categories = [ 'bouncing', 'in_air', 'in_hands', 'custom', 'near_basket' ]

for category in categories: 
    DIR = Path(f'./data/datasets/test/images/{category}')
    filenames = DIR.glob('img*.jpg')

    vid = 10
    get_new_filename = lambda i, video_idx: f'v{vid}_img{i:03d}.jpg'

    for i, filename in enumerate(filenames):
        new_filename = DIR/get_new_filename(i, vid)
        os.rename(filename, new_filename)

    print(f"Renamed {len(list(filenames))} {category} images.")