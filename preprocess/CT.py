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
'''
窗变换
'''
def window_transform(slice,max,min):
    width = float(max) - float(min)
    newimg = (slice-min)/float(width)
    newimg = newimg*255
    return newimg

from PIL import Image
def saveimg(save_path,array):
    im = Image.fromarray(array)
    if im.mode == "F":
        im = im.convert('RGB') 
    im.save(save_path)


def getCT(ct_path,label_path,root_path,label_root_path,number):
    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    # oringin = ct.GetOrigin()
    # direction = ct.GetDirection()
    # xyz_thickness = ct.GetSpacing()
    slices,rows,cols = ct_array.shape
    for k in range(slices):
        target_num = np.sum(label_array[k,:,:] != 0)
        print(target_num)
        if target_num > 100:
            transform_0 = np.where((ct_array[k,:,:]>-850) & (ct_array[k,:,:]<1500),ct_array[k,:,:],0)
            interval = transform_0[transform_0 !=0]
            newlabel = label_array[k,:,:]
            max = np.max(interval)
            min = np.min(interval)
            print('第{}个切片*******************************'.format(k))
            newimg = window_transform(ct_array[k,:,:],max,min)
            save_path = os.path.join(root_path,'liver'+'_'+str(number)+'_'+str(k)+'.png')
            save_label_path = os.path.join(label_root_path,'liver'+'_'+str(number)+'_'+str(k)+'.png')
            saveimg(save_path,newimg)
            saveimg(save_label_path,newlabel)



    # for k in range(slices):
    #     target_interval = []
    #     for i in range(rows):
    #         for j in range(cols):
    #             if((ct_array[k,:,:][i][j] > -850) and (ct_array[k,:,:][i][j] < 1500)):
    #                 target_interval.append(ct_array[k,:,:][i][j])
    #             else:
    #                 ct_array[k,:,:][i][j] = -3024
    #     target_num = np.sum(label_array[k,:,:] != 0)
    #     print(target_num)
    #     if target_num > 10000:
    #         newlabel = np.zeros((rows,cols))
    #         for i in range(rows):
    #             for j in range(cols):
    #                 if((label_array[k,i,j] == 1) or (label_array[k,i,j] == 2)):
    #                     newlabel[i][j] = 255       
    #         max = np.max(target_interval)
    #         min = np.min(target_interval)
    #         print('第{}个切片*******************************'.format(k))
    #         newimg = window_transform(ct_array[k,:,:],max,min)
    #         save_path = os.path.join(root_path,'liver'+'_'+str(number)+'_'+str(k)+'.png')
    #         save_label_path = os.path.join(label_root_path,'seg'+'_'+str(number)+'_'+str(k)+'.png')
    #         saveimg(save_path,newimg)
    #         saveimg(save_label_path,newlabel)




path = 'D://traindata//ISBI_2017_liver//LITS17//'
save_path = 'D://traindata//ISBI_2017_liver//png//'
label_root_path = 'D://traindata//ISBI_2017_liver//png_label//'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(label_root_path):
    os.mkdir(label_root_path)
# processedlist = [0,1,10,100,101,102,103,104,105,106,107]
for name in os.listdir(path):
    number = name[7:-4]
    label_name = 'segmentation-'+number+'.nii'
    # if((name[0] == 'v') and (int(number) not in processedlist)):
    if(name[0] == 'v'):
        ct_path = os.path.join(path,name)
        label_path = os.path.join(path,label_name)
        getCT(ct_path,label_path,save_path,label_root_path,number)











'''
3D
'''
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D           
def plot3D(ct_path):
    print('?')
    ct = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct)
    slices,rows,cols = ct_array.shape
    target_interval = []
    for k in range(slices):
        print('第一阶段',k)
        for i in range(rows):
            for j in range(cols):
                if((ct_array[k,:,:][i][j] > -850) and (ct_array[k,:,:][i][j] < 1500)):
                    target_interval.append(ct_array[k,:,:][i][j])
                else:
                    ct_array[k,:,:][i][j] = -3024
    max = np.max(target_interval)
    min = np.min(target_interval)
    width = float(max) - float(min)
    newimg = np.zeros((slices,rows,cols))
    for k in range(slices):
        print('第二阶段',k)
        for i in range(rows):
            for j in range(cols):
                if(ct_array[k][i][j] == -3024):
                    newimg[k][i][j] = 0
                else:
                    newimg[k][i][j] = (ct_array[k][i][j]-min)/float(width)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(newimg[0], newimg[1], newimg[2], rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    fig = plt.figure()
    ax =fig.add_subplot(projection='3d')
    myax = ax.scatter(newimg[0], newimg[1], newimg[2], maker='.',c=plt.cm.viridis(newimg))
    ax.set_xlim([0.91])
    ax.set_ylim([0.109])
    ax.set_zlim([0.91])
    plt.show()
# plot3D('D://traindata//ISBI_2017_liver//LITS17//volume-1.nii')