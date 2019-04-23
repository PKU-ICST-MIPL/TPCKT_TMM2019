exe="transfer-caffe-2/install/bin/extract_features" #Set your caffe path
model="./modelsaver/last.caffemodel"
prototxt="./model_extract.prototxt"
feature="./CommonFeature"
GPU="GPU 3"

rm -rf ${feature}

mkdir ${feature}
mkdir ${feature}/wiki_img_feature
mkdir ${feature}/wiki_txt_feature
mkdir ${feature}/xmedia_img_feature
mkdir ${feature}/xmedia_txt_feature

$exe $model $prototxt wiki_img_feature ${feature}/wiki_img_feature 34 leveldb $GPU
$exe $model $prototxt wiki_txt_feature ${feature}/wiki_txt_feature 34 leveldb $GPU
$exe $model $prototxt xmedia_img_feature ${feature}/xmedia_img_feature 500 leveldb $GPU
$exe $model $prototxt xmedia_txt_feature ${feature}/xmedia_txt_feature 500 leveldb $GPU
