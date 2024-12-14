import os

folder = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/SHOT/list_file/office/'
domains = os.listdir(folder)
domains.sort()

# print(domains)
# raise Exception
for d in range(len(domains)):
    dom = domains[d]
    if os.path.isdir(os.path.join(folder, dom)):
        dom_new = dom.replace(" ", "_")
        # print(dom, dom_new)
        os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))
        classes = os.listdir(os.path.join(folder, dom_new, 'images'))
        classes.sort()
        # print(classes)
        f = open(folder + dom_new[0] + "_list.txt", "w")
        for c in range(len(classes)):
            cla = classes[c]
            cla_new = cla.replace(" ", "_")
            # print(cla, cla_new)
            os.rename(os.path.join(folder, dom_new, 'images', cla), os.path.join(folder, dom_new, 'images',cla_new))
            files = os.listdir(os.path.join(folder, dom_new, 'images',cla_new))
            files.sort()
            # print(files)
            for file in files:
                file_new = file.replace(" ", "_")
                os.rename(os.path.join(folder, dom_new, 'images',cla_new, file),
                          os.path.join(folder, dom_new, 'images',cla_new, file_new))
                # print(file, file_new)
                print('{:} {:}'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
                f.write('{:} {:}\n'.format(os.path.join(folder, dom_new, 'images',cla_new, file_new), c))
        f.close()
