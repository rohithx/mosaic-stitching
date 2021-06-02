#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)   
    
    ######## SIFT
    sift=cv2.SIFT_create(nfeatures=300)
    kp1,des1=sift.detectAndCompute(gray1,None)
    kp2,des2=sift.detectAndCompute(gray2,None)    
    
    ######## Match
    
    def match(d1, d2, n):

        l1 = len(d1[:,])
        l2 = len(d2[:,])
        l2d = np.empty([l1,l2])    
        
        for i in range(0,l1): 
            ti = d1[i,:]
            for j in range(0,l2):
                tj = d2[j,:]
                t = 0
                for k in range(0,128):
                   t += (ti[k] - tj[k])**2
                l2d[i,j] = t**0.5
        
        idx = np.argpartition(l2d.ravel(), n)
        return l2d,tuple(np.array(np.unravel_index(idx, l2d.shape))[:, range(min(n, 0), max(n, 0))])
        
    l2d, ind = match(des1, des2, 45)  
    
    ######## Homography
    def homography(k1,k2,ind,n):
        
        nkp1 = []
        nkp2 = []
            
        for i in range(0,n):
            nkp1.append(k1[ind[0][i]].pt)
            nkp2.append(k2[ind[1][i]].pt)
        
        h, mask = cv2.findHomography(np.float32(nkp1), np.float32(nkp2), cv2.RANSAC)
        
        return h
    
    h1 = homography(kp1, kp2, ind, 15)
    h2 = np.linalg.inv(h1)   
    
    ######## Stitch        
    def stitch(img, dst, h):
    
        imgh, imgw = img.shape[:2]
        img_bor = np.array([[0, imgw, imgw, 0], [0, 0, imgh, imgh], [1, 1, 1, 1]])
    
        t_img_bor = h.dot(img_bor)
        t_img_bor /= t_img_bor[2,:]
    
        xmin, ymin = np.min(t_img_bor[0,:]), np.min(t_img_bor[1,:])
        xmax, ymax = np.max(t_img_bor[0,:]), np.max(t_img_bor[1,:])
    
        l,b,c = dst.shape
        padsize = [l,b,c] 
        padsize[0] = np.round(np.maximum(l, ymax) - np.minimum(0, ymin)).astype(int)
        padsize[1] = np.round(np.maximum(b, xmax) - np.minimum(0, xmin)).astype(int)
        dst_fin = np.zeros(padsize, dtype=np.uint8)
    
        transx, transy = 0, 0
        transh = np.eye(3,3)
        if xmin < 0: 
            transx = np.round(-xmin).astype(int)
            transh[0,2] += transx
        if ymin < 0:
            transy = np.round(-ymin).astype(int)
            transh[1,2] += transy
            
        newh = transh.dot(h)
        newh /= newh[2,2]
    
        dst_fin[transy:transy+l, transx:transx+b] = dst
    
        warp_fin = cv2.warpPerspective(img, newh, (padsize[1],padsize[0]))
    
        return dst_fin, warp_fin
    
    timg2, timg1 = stitch(img2, img1, h2)
    
    # Blending
    def blend(imx, imy, thresh):
        if imx.shape == imy.shape:
            new_blend = np.zeros(imx.shape, dtype=np.uint8)
            for i in range(0,imx.shape[1]):
                for j in range(0,imx.shape[0]):
                    v1 = np.linalg.norm(imx[j,i,:])
                    v2 = np.linalg.norm(imy[j,i,:])
                    x = np.abs(v1 - v2)
                    if v1 == 0 and v2 != 0:
                        new_blend[j,i,:] = imy[j,i,:]
                    elif v1 != 0 and v2 == 0:
                        new_blend[j,i,:] = imx[j,i,:]
                    elif v1 != 0 and v2 != 0:
                        if x < thresh:
                            new_blend[j,i,:] = cv2.addWeighted(imx[j,i,:], 0.5, imy[j,i,:], 0.5, 1.0).reshape(3,)
                        else:
                            if i < imx.shape[1]/2:
                                new_blend[j,i,:] = imx[j,i,:]
                            else:
                                new_blend[j,i,:] = imy[j,i,:]
        else:
            print('incorrect dimensions')
        return new_blend
    
    b = blend(timg1, timg2, 10)    
    
    ######## Save
    cv2.imwrite(savepath, b)    

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

