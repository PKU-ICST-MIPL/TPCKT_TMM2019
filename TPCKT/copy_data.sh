cp -f ./TrainData/label/train_img.txt ./TrainData/label/new_train_img.txt

rm -rf ./TrainData/XmediaData
mkdir ./TrainData/XmediaData
rm -rf ./TrainData/WikiData
mkdir  ./TrainData/WikiData
cp -r /xmedianet_train_pool5/ ./TrainData/XmediaData  #TrainData_XMediaNet_Image in LMDB
cp -r /lmdb_train/ ./TrainData/XmediaData				 #TrainData_XMediaNet_Text in LMDB
cp -r /wiki_train_pool5 ./TrainData/WikiData/wiki_train_pool5 #TrainData_Wikipedia_Image in LMDB
cp -r /lmdb_train ./TrainData/WikiData/lmdb_train			  #TrainData_Wikipedia_Text in LMDB
