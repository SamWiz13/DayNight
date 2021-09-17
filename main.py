import os
import glob
import cv2
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import  numpy as np

training_data_dir ="tun_kun/training/"
test_data_dir ="tun_kun/test/"

# Datasetni load qilamiz
def Load_dataset(data_dir):
    img_list =[]
    img_types =['day','night']
    for i in img_types:
        for fayl in glob.glob(os.path.join(data_dir,i,'*')):
            img =mimg.imread(fayl)
            if not img is None:
                img_list.append((img,i))
    return img_list
imgs =Load_dataset(training_data_dir)

# Kun yoki tunlikka tekshiramiz
def Read():
    for i in range(len(imgs)):
        if imgs[i][1] == 'night':
            plt.title(imgs[i][1])
            plt.imshow(imgs[i][0])
            plt.show()
            break

# Rasmlarni bir xil xajimga keltiramiz
def Standart(imgs):
        res_imgs = cv2.resize(imgs, (660, 480))
        return res_imgs

# Labelini encode qilamiz
def Encode(label):
        encode_label =0
        if label =='day':
           encode_label =1
        return encode_label

# Rasmlarni listga yig'amiz
def Standart_list(imgs):
    standart_imgs = []
    for i in range(len(imgs)):
        standart_imgs.append((Standart(imgs[i][0]),Encode(imgs[i][1])))
    return standart_imgs

standart_list =Standart_list(imgs)

# Yorug'likning o'rtacha qiymatini topamiz
def Average(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    return np.sum(hsv[:,:,2]) / (480*660)

# Datasetni test qilamiz
test_imgs =Load_dataset(test_data_dir)
standart_test =Standart_list(test_imgs)

# Rasmarni aniqlaymiz
def Identify(imgs,night_angels):
    if np.sum(imgs) >1.88*night_angels:
        return 1
    return 0

# Rasm xatolarini topamiz
def Accuracy(imgs):
    mistake =[]
    img =[np.sum(i[0]) for i in imgs]
    average =sum(img) /len(img)
    img =list(map(lambda x:x-average,img))
    img =list(map(lambda x:x*x,img))
    night_angels =np.sqrt(sum(img) /len(img))
    for i in imgs:
        img = i[0]
        label = i[1]

        predict_label = Identify(img,night_angels)
        if predict_label != label:
            mistake.append((img, label, predict_label))
    return mistake

mistake =Accuracy(standart_test)
general =len(standart_test)
detail =general -len(mistake)
degree =detail /general
print("Daraja :",str(degree*100))
print("Mistake image :"+str(len(mistake)) +" out of " + str(general))

# Xato rasmlar
def Mistkes():
    plt.figure(figsize=(16, 16))
    for i in range(len(mistake)):
            plt.subplot(2, 6, i + 1)
            plt.title(str(mistake[i][1]) + "-" + str(mistake[i][2]))
            plt.imshow(mistake[i][0])
    return plt.show()
imgs2 =Mistkes()








