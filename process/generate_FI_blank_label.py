import os

emo_dict = {
    'amusement': 0,
    'anger': 1,
    'awe': 2,
    'contentment': 3,
    'disgust': 4,
    'excitement': 5,
    'fear': 6,
    'sadness': 7
}

path = '/nfs/ofs-902-1/object-detection/zhujiankun/EmotionDA/FI_full/FI'
for emotion in os.listdir(path):
    print(emotion)
    if emotion == '.DS_Store':
        continue
    num = emo_dict[emotion]
    emo_path = os.path.join(path,emotion)
    count = 0
    for pic in os.listdir(emo_path):
        count += 1
        if count > 100:
            break
        full_path = os.path.join(emo_path,pic) + ' ' + str(num) + '\n'
        with open(f'/nfs/ofs-902-1/object-detection/zhujiankun/EmotionDA/FI_small_for_tsne/label_100/{emotion}.txt','a')as file:
            file.write(full_path)

