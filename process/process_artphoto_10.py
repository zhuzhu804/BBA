import random
from sklearn.model_selection import train_test_split

# Path to your text file containing image paths
file_path = '../list_file/Art_FI_10/Art_full.txt'

# Read image paths from the file
with open(file_path, 'r') as file:
    image_paths = file.read().splitlines()

# Repeat the random split 10 times
for i in range(10):
    # Split the data into 80% train and 20% test
    train, test = train_test_split(image_paths, test_size=0.2, random_state=i)

    # Save the splits to new files
    with open(f'train_split_{i}.txt', 'w') as file:
        for item in train:
            file.write("%s\n" % item)

    with open(f'test_split_{i}.txt', 'w') as file:
        for item in test:
            file.write("%s\n" % item)

    print(f'Split {i+1} done. Train size: {len(train)}, Test size: {len(test)}')
