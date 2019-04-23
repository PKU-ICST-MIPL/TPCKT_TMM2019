# Introduction
This is the source code our TMM 2019 paper "TPCKT: Two-level Progressive Cross-media Knowledge Transfer".

Xin Huang and Yuxin Peng, "TPCKT: Two-level Progressive Cross-media Knowledge Transfer", IEEE Transactions on Multimedia (TMM), DOI:10.1109/TMM.2019.2911456, 2019. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=20199)

# Content
1. XMediaNet: Source model on XMediaNet dataset (for pre-training)
2. Wiki: Target model on Wikipedia dataset (for pre-training), which also has model.prototxt, solver.prototxt, and test.prototxt
3. TPCKT: codes of training and testing for TPCKT
4. VGG19: Pre-train model of VGG19, please download this [model](http://59.108.48.34/tiki/tiki-download_file.php?fileId=10051)
5. Evaluate: Test codes for generating MAP scores

# Usage
1. Setup transfer-caffe <br/>
Setup transfer-caffe from the following URL: https://github.com/zhuhan1236/transfer-caffe

2. Pre-training for Source and Target model
* Training of Source model in folder "XMediaNet", as SourceModel.caffemodel. According to model.prototxt, you need:
  * Extracting the pool5 feature maps of XMediaNet dataset, as .LMDB format, using vgg19_cvgj_iter_300000.caffemodel and test.prototxt in folder "VGG19". <br/>
	  You need images' folder, and list in .txt format (including label). Remember to set your paths in test.prototxt.  
		Each line of List is in the format as "filepath label" like "n04347754_15004.JPEG 833" <br/>
  * Extracting the text features, in .LMDB format. In our paper, each text is represented as a 300-d Word CNN feature. <br/>
	  For .LMDB format, each entry of lmdb includes this vector and its label. <br/>
  * Training source model as SourceModel.caffemodel. Use solver.prototxt and model.prototxt, with pre-train model vgg19_cvgj_iter_300000_TripleNet.caffemodel. <br/>
	   Remember to set your paths.
* Training of Target model in folder "Wiki", as TargetModel.caffemodel. Similar to the Source model. <br/>
	XMediaNet dataset can be download via http://www.icst.pku.edu.cn/mipl/XmediaNet <br/>
	Wikipeia dataset can be download via: http://www.svcl.ucsd.edu/projects/crossmodal/ <br/>

3. Progressive Transfer
* Prepare the setting in folder "TPCKT", including:
  * Put the training data LIST for XMeidaNet dataset in "TrainData/labelas" as train_img.txt, and train_txt.txt. <br/>
		Put the training data LIST for Wikipedia dataset in "TrainData/labelas" as imageTrainList.txt, and textTrainList.txt. <br/>
		Each line of List is in the format as "filepath label" like "n04347754_15004.JPEG 833" <br/>
  * Set your data path in: <br/>
	  copy_data.sh <br/>
		solver_Full_wiki.prototxt <br/>
		solver_new.prototxt <br/>
		model_extract.prototxt <br/>
		model_grl.prototxt <br/>
		Do not change the paths with "TrainData...", which are actually temp files. <br/>
  * Set your caffe path in all .sh file.
* Run progressive_train.sh

Note: The most common problem may come from the path settings. I suggest first dealing with TargetModel and SourceModel respectively, and then only one 1 Iter of TPCKT, and so on.


4. Evaluation
* If progressive_train.sh is succesfully run, the common representations can be found in folder "Feature"
* Compute MAP scores with extracted representations with Evaluate/evaluate_wiki.m. Note: We set an exapmle Label.mat file in this folder. You must create yourselves to match the labels of your test data.
