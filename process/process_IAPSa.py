from scipy.io import loadmat

# Load the .mat file
mat_data = loadmat('/Users/didi/Desktop/IAPSa/LabelsIAPS.mat')

# Display the keys to understand the structure of the data
print(mat_data.keys())

# Extracting and displaying the content for each key (excluding '__header__', '__version__', '__globals__')

# To avoid displaying large amounts of data, we'll summarize the content of each key
key_summaries = {}
labels = []
for key in mat_data.keys():
    if key not in ['__header__', '__version__', '__globals__']:
        content = mat_data[key].tolist()
for index in range(len(content)):
    index_list = []
    for key in mat_data.keys():
        if key not in ['__header__', '__version__', '__globals__', 'labels']:
            content_idx = mat_data[key].tolist()[index]
            index_list.append(content_idx)
    labels.append(index_list)


print(len(labels))
print(len(labels[0]))
print(labels)

with open('/Users/didi/Desktop/IAPSa/ImageNames.txt','r')as image_name:
    image_names = image_name.readlines()
    for idx in range(len(image_names)):
        name = image_names[idx].strip()
        label = labels[idx]
        lst_flattened = [item for sublist in label for item in sublist]
        str_result = ' '.join(map(str, lst_flattened))
        pic_label = name + ' ' + str_result + '\n'
        with open('/Users/didi/Desktop/IAPSa/labels.txt','a')as label_file:
            label_file.write(pic_label)
