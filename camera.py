import numpy as np, cv2
import Filter as Ft
import Functions as Fu
import matplotlib.pyplot as plt
import getFingerprint as gF
import extraUtils as eu
import maindir as md
import os
import scipy.io as sio
from tqdm.auto import tqdm

im6='data' + os.sep + '/data1.jpg'
im7='data' + os.sep + '/data2.jpg'
im8='data' + os.sep + '/data3.jpg'
im9='data' + os.sep + '/t.jpg'
im10='data' + os.sep + '/data5.jpg'
im11='data' + os.sep + '/data6.jpg'
im12='data' + os.sep + '/data7.jpg'
im13='data' + os.sep + '/data8.jpg'
im14='data' + os.sep + '/data10.jpg'
im15='data' + os.sep + '/data11.jpg'
im16='data' + os.sep + '/data12.jpg'
im17='data' + os.sep + '/data9.jpg'
im18='data' + os.sep + '/data13.jpg'
im19='data' + os.sep + '/data14.jpg'
im20='data' + os.sep + '/data15.jpg'
im21='data' + os.sep + '/data16.jpg'
im22='data' + os.sep + '/data17.jpg'
im23='data' + os.sep + '/data18.jpg'
im24='data' + os.sep + '/data19.jpg'
im25='data' + os.sep + '/data20.jpg'
im26='data' + os.sep + '/data21.jpg'

Images = [im6,im7,im8,im9,im10,im11,im12,im13,im14,im15,im17,im18,im19,im20,im21,im22,im23,im24,im25,im26]

# RP,_,_ = gF.getFingerprint(Images)  #Fingerprint 가 이미지 쉐입만큼 나온다.
# print(RP.shape)
# RP1 ,RP2,RP3=RP[:,:,0],RP[:,:,1],RP[:,:,2],
# sigmaRP1, sigmaRP2, sigmaRP3 = np.std(RP1),np.std(RP2),np.std(RP3)
# Fingerprint1 ,Fingerprint2,Fingerprint3 = Fu.WienerInDFT(RP1, sigmaRP1),Fu.WienerInDFT(RP2, sigmaRP2), Fu.WienerInDFT(RP3, sigmaRP3)
# RP=np.array([RP1,RP2,RP3])
# sigmaRP=np.array([sigmaRP1,sigmaRP2,sigmaRP3])
# Fingerprint=np.array([Fingerprint1,Fingerprint2,Fingerprint3])
#
# sio.savemat('Fingerprint.mat', {'RP': RP, 'sigmaRP': sigmaRP, 'Fingerprint': Fingerprint})

mat_file = sio.loadmat("Fingerprint.mat")
Fingerprint= mat_file["Fingerprint"]


def huechange(img, shift):
    #plt.imshow(img)
    #plt.show()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #print(hsv[:, :, 0])
    hsv[:, :, 0]=np.where(hsv[:, :, 0]>=180-shift,hsv[:, :, 0] -180+shift,hsv[:, :, 0] +shift)
    #print(hsv[:, :, 0])
    hnew = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # plt.imshow(hnew)
    # plt.show()
    return hnew

def Corr(imm):
    result = []
    for i in tqdm(range(0, 181,20)):
        # hnew[:,:,0]+=i
        #im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        # hnew1 = cv2.cvtColor(hnew1, cv2.COLOR_HSV2BGR)
        hnew1 = huechange(imm, i)
        print("################## ",i ,"Shift #####################")
        Noisex = Ft.NoiseExtractFromImage(hnew1, sigma=2.)
        PCESUM = 0
        for j in range(3):
            Noisex2 = Fu.WienerInDFT(Noisex[:,:,j], np.std(Noisex[:,:,j]))
            C = Fu.crosscorr(Noisex2,np.multiply(hnew1[:,:,j],Fingerprint[j]))
            det, _ = md.PCE(C)
            PCESUM += (det["PCE"])
        result.append(PCESUM)

    return result


img = cv2.imread('./data/coke.jpg')
im = img[:,:,::-1]
# im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

gak=160
hhh=huechange(im,gak)

result=Corr(hhh)
idx=np.argmax(np.array(result))

print("############detect###########")
print("PCESUM MAX INDEX :",idx)
if idx == 0:
    print("no hue modified")
else:
    print("modified")

changeimg = huechange(hhh, idx*20)

plt.subplot(1, 3, 1), plt.title("original"), plt.imshow(im)
plt.subplot(1, 3, 2), plt.title("shift"), plt.imshow(hhh)
plt.subplot(1, 3, 3), plt.title("foundimage"), plt.imshow(changeimg)
plt.show()



