import scipy.io as scio

path = '/Users/didi/Desktop/e04728dd246b7460cf47ab142a2dbd0c.mat'
matdata = scio.loadmat(path)
print(matdata)