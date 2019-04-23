#coding=utf-8

import numpy as np
import lmdb
import time
import os
from sys import argv

def calc_dist(f):
	if np.linalg.norm(f) == 0:
		return -10
	num = float(0)
	for g in wiki_category:
		num += np.dot(f, g)/float(np.linalg.norm(f) * np.linalg.norm(g))
	return num

def calc_dist_wiki(f):
	if np.linalg.norm(f) == 0:
		return -200
	num = float(0)
	for g in xm_category:
		num += np.dot(f, g)/float(np.linalg.norm(f) * np.linalg.norm(g))
	return num

wiki_img = np.loadtxt("./CommonFeature/wiki_img_feature/feature.txt")
wiki_txt = np.loadtxt("./CommonFeature/wiki_txt_feature/feature.txt")
print("Loading wiki features done.")

xm_img = np.loadtxt("./CommonFeature/xmedia_img_feature/feature.txt")
xm_txt = np.loadtxt("./CommonFeature/xmedia_txt_feature/feature.txt")
print("Loading xmedia features done.")
'''
np.savetxt("../CommonFeature/xmedia_img_feature/feature_part.txt", xm_img[:1000])
np.savetxt("../CommonFeature/xmedia_txt_feature/feature_part.txt", xm_txt[:1000])
print("Creating part of xmedia features done.")

xm_img = np.loadtxt("./CommonFeature/xmedia_img_feature/feature_part.txt")
xm_txt = np.loadtxt("./CommonFeature/xmedia_txt_feature/feature_part.txt")
print("Loading part of xmedia features done.")
'''
file_wiki_label = open('./TrainData/label/textTrainList.txt')
lines = file_wiki_label.readlines()
file_wiki_label.close()
wiki_full_label = []
for tempstring in lines:
	tempstring = tempstring.split()
	wiki_full_label.append(tempstring)
wiki_label = np.array([x[-1] for x in wiki_full_label]).astype(int)
print("Loading wiki label done.")

file_xm_label = open('./TrainData/label/train_txt.txt')
lines = file_xm_label.readlines()
file_xm_label.close()
xm_full_label = []
for tempstring in lines:
	tempstring = tempstring.split()
	xm_full_label.append(tempstring)
xm_label = np.array([x[-1] for x in xm_full_label]).astype(int)
print("Loading xmedia label done.")

#xm_label = xm_label[:1280]

wiki_category = np.zeros((10, 4096))
xm_category = np.zeros((200, 4096))
wiki_num = np.zeros(10)
xm_num = np.zeros(200)

for i in range(wiki_label.shape[0]):
	wiki_num[wiki_label[i]] += 2
	wiki_category[wiki_label[i]] += wiki_img[i]
	wiki_category[wiki_label[i]] += wiki_txt[i]

for i in range(wiki_category.shape[0]):
	wiki_category[i] /= wiki_num[i]

for i in range(xm_label.shape[0]):
	xm_num[xm_label[i]] += 2
	xm_category[xm_label[i]] += xm_img[i]
	xm_category[xm_label[i]] += xm_txt[i]

for i in range(xm_category.shape[0]):
	if xm_num[i] != 0:
		xm_category[i] /= xm_num[i]

xm_dist = [calc_dist(f) for f in xm_category]
wiki_dist = [calc_dist(f) for f in wiki_category]

xm_sorted_label = np.argsort(xm_dist) #倒序排，排在前面的是需要删除的
wiki_sorted_label = np.argsort(wiki_dist) #倒序排，排在前面的是需要删除的

#print(xm_sorted_label)
Iter = int(argv[1])/10
print(Iter)
xm_threshold = int((0.2 ** Iter) * xm_sorted_label.shape[0])
wiki_threshold = int((0.2 ** Iter) * wiki_sorted_label.shape[0])

xm_Deleted_label = xm_sorted_label[:xm_threshold]
wiki_Deleted_label = wiki_sorted_label[:wiki_threshold]

print("Xmedia Remained label", xm_sorted_label[xm_threshold:])
print("Wiki Remained label", wiki_sorted_label[wiki_threshold:])

#time.sleep(3000)
#os.system("rm -r ./TrainData/XmediaData/xmedianet_train_pool5_2x")
#os.system("rm -r ./TrainData/XmediaData/lmdb_train")
#os.system("rm -r ./TrainData/WikiData/wiki_train_pool5")
#os.system("rm -r ./TrainData/WikiData/lmdb_train/")
#os.system("cp -r /home/yuanmingkuan/huangxin/CVPR21.0/Pool5Generator/Feature/xmedianet_train_pool5_2x ./TrainData/XmediaData/xmedianet_train_pool5_2x")
#os.system("cp -r /home/yuanmingkuan/huangxin/CVPR21.0/DataGenerator/WCNN/XMedia/lmdb_train ./TrainData/XmediaData/lmdb_train")
#os.system("cp -r /home/yuanmingkuan/huangxin/CVPR21.0/Pool5Generator/Feature/wiki_train_pool5 ./TrainData/WikiData/wiki_train_pool5")
#os.system("cp -r /home/yuanmingkuan/huangxin/CVPR21.0/DataGenerator/WCNN/Wiki/lmdb_train ./TrainData/WikiData/lmdb_train/")
xm_lmdb_img = lmdb.open('./TrainData/XmediaData/xmedianet_train_pool5/', map_size=int(1e12))
xm_lmdb_txt = lmdb.open('./TrainData/XmediaData/lmdb_train/', map_size=int(1e12))
wiki_lmdb_img = lmdb.open('./TrainData/WikiData/wiki_train_pool5/', map_size=int(1e12))
wiki_lmdb_txt = lmdb.open('./TrainData/WikiData/lmdb_train/', map_size=int(1e12))

xm_txn_img = xm_lmdb_img.begin(write = True)
xm_txn_txt = xm_lmdb_txt.begin(write = True)
wiki_txn_img = wiki_lmdb_img.begin(write = True)
wiki_txn_txt = wiki_lmdb_txt.begin(write = True)




for i, label in enumerate(xm_full_label):  #   XmediaNet, Write
	if int(label[1]) in xm_Deleted_label:
		xm_key_txt = str(i).zfill(8) + '_' + label[0]
		#print(xm_key_txt)
		xm_txn_txt.delete(xm_key_txt)
		xm_key_img = str(i).zfill(10)
		#print(xm_key_img)
		xm_txn_img.delete(xm_key_img)
		#time.sleep(5)


Counter = 0
for key, value in xm_txn_img.cursor():
        Counter = Counter + 1
print (Counter)

Counter = 0
for key, value in xm_txn_txt.cursor():
        Counter = Counter + 1
print (Counter)


xm_txn_img.commit()
xm_txn_txt.commit()
xm_lmdb_img.close()
xm_lmdb_txt.close()




for i, label in enumerate(wiki_full_label):  #   Wiki, WriteFeature
	if int(label[1]) in wiki_Deleted_label:
		wiki_key_txt = str(i).zfill(8) + '_' + label[0]
		wiki_txn_txt.delete(wiki_key_txt)
		wiki_key_img = str(i).zfill(10)
		wiki_txn_img.delete(wiki_key_img)

Counter = 0
for key, value in wiki_txn_img.cursor():
        Counter = Counter + 1
print (Counter)

Counter = 0
for key, value in wiki_txn_txt.cursor():
        Counter = Counter + 1
print (Counter)



wiki_txn_img.commit()
wiki_txn_txt.commit()

wiki_lmdb_img.close()
wiki_lmdb_txt.close()


xm_img_traininglist = open('./TrainData/label/train_img.txt')   #XmediaNet, WriteList
lines = xm_img_traininglist.readlines()
xm_img_traininglist.close()
xm_newtraininglist = []
for tempstring in lines:
	tempstring = tempstring.split()
	if int(tempstring[1]) in xm_Deleted_label:
		continue
	xm_newtraininglist.append(tempstring)

xm_file_newtraininglist = open('./TrainData/label/new_train_img.txt', 'w')

for key, value in xm_newtraininglist:
	print >>xm_file_newtraininglist, key + ' ' + value

xm_file_newtraininglist.close()


wiki_img_traininglist = open('./TrainData/label/imageTrainList.txt')   #Wiki, WriteList
lines = wiki_img_traininglist.readlines()
wiki_img_traininglist.close()
wiki_newtraininglist = []
for tempstring in lines:
	tempstring = tempstring.split()
	if int(tempstring[1]) in wiki_Deleted_label:
		continue
	wiki_newtraininglist.append(tempstring)

wiki_file_newtraininglist = open('./TrainData/label/Wiki_new_train_img.txt', 'w')

for key, value in wiki_newtraininglist:
	print >>wiki_file_newtraininglist, key + ' ' + value

wiki_file_newtraininglist.close()
