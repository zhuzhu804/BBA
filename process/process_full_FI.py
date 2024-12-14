emotion_dict_l2n = {
    1: 'Amusement',
    2: 'Contentment',
    3: 'Awe',
    4: 'Excitement',
    5: 'Fear',
    6: 'Sadness',
    7: 'Disgust',
    8: 'Anger'
}

emotion_dict_n2l = {
    'Amusement': 0,
    'Anger': 1,
    'Awe': 2,
    'Contentment': 3,
    'Disgust': 4,
    'Excitement': 5,
    'Fear': 6,
    'Sadness': 7
}

root = '/nfs/ofs-902-1/object-detection/zhujiankun/EmotionDA/FI_full/FI/Images/'
with open('./TrainImages.txt', 'r') as test_file:
    all_file = test_file.readlines()
    for pic in all_file:
        # print(pic)
        label, pic_name = int(pic.strip().split('/')[0]), pic.strip().split('/')[1]
        re_name = root + emotion_dict_l2n[label] + '/' + pic_name + ' ' + str(emotion_dict_n2l[emotion_dict_l2n[label]]) + '\n'
        with open('./FI_train.txt','a')as test_FI:
            test_FI.write(re_name)
