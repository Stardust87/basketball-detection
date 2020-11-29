import glob
import shutil, os

OUTPUT_PATH = '.\\data\\CAMERA\\train\\output'
FROM_PATH = '.\\data\\CAMERA\\train\\images'

filenames_to = glob.glob(f'{OUTPUT_PATH}\\*.jpg')
filenames_from = [ FROM_PATH+'\\'+f.split('\\')[-1] for f in filenames_to]

for key, f in enumerate(filenames_from):
    shutil.copyfile(f, filenames_to[key])
    os.remove(f)