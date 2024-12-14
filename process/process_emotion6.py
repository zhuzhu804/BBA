import os

from pathlib import Path

directory = '/nfs/ofs-902-1/object-detection/zhujiankun/EmotionDA/Emotion6/Emotion-6/images'
with open('/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/SHOT/list_file/Emotion6_FI/Emotion6_full.txt', 'a')as Emo_full:
    path = Path(directory)
    for file in path.rglob('*'):
        if file.is_file():
            print(file)
            Emo_full.write(str(file) + '\n')


