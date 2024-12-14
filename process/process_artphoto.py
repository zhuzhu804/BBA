train_file = '/Users/didi/Desktop/EmotionDA/ArtPhoto/split/train.txt'
test_file = '/Users/didi/Desktop/EmotionDA/ArtPhoto/split/test.txt'
root_path = '/nfs/ofs-902-1/object-detection/zhujiankun/EmotionDA/ArtPhoto/testImages_artphoto/'

with open(test_file,'r')as train:
    train_list = train.readlines()
    for train_pic in train_list:
        train_pic = root_path + train_pic
        with open('../list_file/Art_FI/Art_test.txt','a')as train_w:
            train_w.write(train_pic)
