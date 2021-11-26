import os
from tqdm import tqdm
import SimpleITK as sitk
import skimage.io as io
import numpy as np
np.set_printoptions(threshold=np.inf)

#显示图片
def show_img_single(ori_img):
    io.imshow(ori_img, cmap = 'gray')
    io.show()

def window_transform(slice,max,min):
    width = float(max) - float(min)
    rows,cols = slice.shape
    newimg = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if(slice[i][j] == -3024):
                newimg[i][j] = 0
            else:
                newimg[i][j] = (slice[i][j]-min)/float(width)
    newimg = newimg*255
    show_img_single(newimg)
            


def getCT(ct_path,label_path):
    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    oringin = ct.GetOrigin()
    direction = ct.GetDirection()
    xyz_thickness = ct.GetSpacing()
    slices,rows,cols = ct_array.shape
 
    for k in range(slices):
        target_interval = []
        for i in range(rows):
            for j in range(cols):
                if((ct_array[k,:,:][i][j] > -850) and (ct_array[k,:,:][i][j] < 1500)):
                    target_interval.append(ct_array[k,:,:][i][j])
                else:
                    ct_array[k,:,:][i][j] = -3024
        print('有标签的值',np.sum(label_array[k,:,:] != 0))
        # if percentage > 0.2:
        max = np.max(target_interval)
        min = np.min(target_interval)
        print('我进来了*******************************',k)
        #window = window_transform(ct_array[k,:,:],max,min)




path = 'D://traindata//ISBI_2017_liver//LITS17//'
for name in os.listdir(path):
    number = name[7:-4]
    label_name = 'segmentation-'+number+'.nii'
    if(name[0] == 'v'):
        ct_path = os.path.join(path,name)
        label_path = os.path.join(path,label_name)
        getCT(ct_path,label_path)


