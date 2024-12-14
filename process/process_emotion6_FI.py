import os

with open('../list_file/Emotion6_FI/Emotion6_test.txt','r')as origin_FI:
    all_FI = origin_FI.readlines()

    with open('../list_file/Emotion6_FI/E6_test.txt','a')as two_FI:
        for FI_line in all_FI:
            FI_label = FI_line.strip().split('/')[-4]
            if FI_label in ['love', 'joy', 'surprise']:
                new_label = 0
                FI_new_line = FI_line.strip().split(' ')[0] + ' ' + str(new_label) + '\n'
            elif FI_label in ['anger', 'fear', 'sadness']:
                new_label = 1
                FI_new_line = FI_line.strip().split(' ')[0] + ' ' + str(new_label) + '\n'
            two_FI.write(FI_new_line)

# from sklearn.model_selection import train_test_split
#
# # Path to your text file containing image paths
# file_path = '../list_file/Emotion6_FI/Emotion6_full.txt'
#
# # Read image paths from the file
# with open(file_path, 'r') as file:
#     image_paths = file.read().splitlines()
#
# # Repeat the random split 10 times
#
#     # Split the data into 80% train and 20% test
# train, test = train_test_split(image_paths, test_size=0.2)
#
# # Save the splits to new files
# with open(f'Emotion6_train.txt', 'w') as file:
#     for item in train:
#         file.write("%s\n" % item)
#
# with open(f'Emotion6_test.txt', 'w') as file:
#     for item in test:
#         file.write("%s\n" % item)

