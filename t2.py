# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    kp  = []
    des = []
    
    #SIFT
    for i in imgs:
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        sift=cv2.SIFT_create(nfeatures=200)
        kpx,desx=sift.detectAndCompute(gray,None)
        kp.append(kpx)
        des.append(desx)
        
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
        
    
    def check_matches(desc):
        im = np.eye(len(des))
        lf2d = []
        idx = []
        num = []
        ind1 = 0
        for j in desc:
            ind2 = 0
            for k in desc:
                l2d, idn = match(k,j,8)
                idx.append(idn)
                lf2d.append(l2d)
                t = len(np.where(l2d < 100)[0])
                num.append(t)
                if t != 0:
                    im[ind1,ind2] = 1
                ind2 += 1
            ind1 += 1
        return lf2d, idx, num, im
    
    x,w,y,z = check_matches(des)   
        
    def homography(k1,k2,ind,n):
    
        nkp1 = []
        nkp2 = []
            
        for i in range(0,n):
            nkp1.append(k1[ind[0][i]].pt)
            nkp2.append(k2[ind[1][i]].pt)
        
        h, mask = cv2.findHomography(np.float32(nkp1), np.float32(nkp2), cv2.RANSAC)
        
        return h
    
    xxx = np.zeros((len(imgs), len(imgs)))
    t = {}
    
    for i in range(0,len(imgs)):
        for j in range(0,len(imgs)):
            if z[i,j] == 0:
                xxx[i,j] = None
            else:
                h = homography(kp[j],kp[i],w[i*len(imgs)+j],8)
                t[i,j] = h
                
    def stich(imat, hmat, imgs):
        inx = []
        hnx = []
        for i in range(0,len(imgs)):
            for j in range(0,len(imgs)):
                if i > j and imat[i,j] == 1:
                    inx.append([i,j])
                    hnx.append(t[i,j])
        return inx, hnx
                    
    ib1,hn1 = stich(z,t,imgs)
    
    ic = [0 for i in ib1]
    for i in range(0,len(ib1)):
        for j in ib1:
            if i in j:
                ic[i] += 1
                
    tarind = np.argmax(ic)
     
    n = {}
    for j in range(0, len(imgs)):
        for i in range(0,len(ib1)):
            if ib1[i][0] == tarind and ib1[i][1] == j:
                n[j] = hn1[i]
            elif ib1[i][1] == tarind and ib1[i][0] == j:
                n[j] = np.linalg.inv(hn1[i])
    
    dst = imgs[tarind]
    minv = []
    maxv = []
    for i in list(n.keys()):
        img = imgs[i]
        h = n[i]
        imgh, imgw = img.shape[:2]
        img_bor = np.array([[0, imgw, imgw, 0], [0, 0, imgh, imgh], [1, 1, 1, 1]])
    
        t_img_bor = h.dot(img_bor)
        t_img_bor /= t_img_bor[2,:]
    
        xmin, ymin = np.min(t_img_bor[0,:]), np.min(t_img_bor[1,:])
        xmax, ymax = np.max(t_img_bor[0,:]), np.max(t_img_bor[1,:])
        minv.append([xmin,ymin])
        maxv.append([xmax,ymax])
    
    xmin = np.min(minv,0)[0]
    ymin = np.min(minv,0)[1]
    xmax = np.max(maxv,0)[0]
    ymax = np.max(maxv,0)[1]
        
        
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
    
    dst_fin[transy:transy+l, transx:transx+b] = dst
    wrp = []
    for i in list(n.keys()):
        img = imgs[i]
        h = n[i]    
            
        newh = transh.dot(h)
        newh /= newh[2,2]
    
        warp_fin = cv2.warpPerspective(img, newh, (padsize[1],padsize[0]))
        wrp.append(warp_fin)
    
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
                        new_blend[j,i,:] = cv2.addWeighted(imx[j,i,:], 0.5, imy[j,i,:], 0.5, 1.0).reshape(3,)
                        '''
                        else:
                            if i < imx.shape[1]/2:
                                new_blend[j,i,:] = imx[j,i,:]
                            else:
                                new_blend[j,i,:] = imy[j,i,:]'''
        else:
            print('incorrect dimensions')
        return new_blend
    
    curr_img = dst_fin
    for i in wrp:
        b = blend(curr_img, i, 50)
        curr_img = b
    
    overlap_arr = z
    cv2.imwrite(savepath, b)  
    
    return overlap_arr

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
