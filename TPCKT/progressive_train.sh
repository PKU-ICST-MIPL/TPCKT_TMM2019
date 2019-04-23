sudo rm -r Feature/*
exe="transfer-caffe-2/install/bin/caffe" #Set your caffe path
model1="TargetModel.caffemodel"
model2="SourceModel.caffemodel"
solver="./solver_new.prototxt"
gpu="3"

sh copy_data.sh
$exe train --solver="./solver_Full_wiki.prototxt" --weights=$model2,$model1 --gpu=$gpu

cp -f /modelsaver/_iter_1.solverstate ./modelsaver/last.solverstate
cp -f /modelsaver/_iter_1.caffemodel      ./modelsaver/last.caffemodel

sh extract_common.sh
mkdir Feature/1
transfer-caffe-2/install/bin/extract_features ./modelsaver/last.caffemodel ./test_grl.prototxt img_prob ./Feature/1/wiki_img_prob 11 leveldb GPU ${gpu}
transfer-caffe-2/install/bin/extract_features ./modelsaver/last.caffemodel ./test_grl.prototxt txt_prob ./Feature/1/wiki_txt_prob 11 leveldb GPU ${gpu}


for iter in `seq  2  10 100`
do
	printf "iter: %d\n" ${iter}
	read -p "iter"
	sh copy_data.sh
	python py/generate_traininglist.py $((iter-1))
#$	return
	printf "Generating data done.\n"
	#generate new solver
	rm solver_new.prototxt
	cp solver.prototxt solver_new.prototxt
	sed -i "s/MAXITER/$iter/" solver_new.prototxt
	
	model="./modelsaver/last.solverstate"
	$exe train --solver=$solver --snapshot=$model --gpu=$gpu
	
	cp -f /modelsaver/_iter_${iter}.solverstate	./modelsaver/last.solverstate
	cp -f /modelsaver/_iter_${iter}.caffemodel	./modelsaver/last.caffemodel

    sh extract_common.sh
    mkdir Feature/$((iter))
    /home/yuanmingkuan/transfer-caffe-2/install/bin/extract_features ./modelsaver/last.caffemodel ./test_grl.prototxt img_prob ./Feature/$((iter))/wiki_img_prob 11 leveldb GPU ${gpu}
    /home/yuanmingkuan/transfer-caffe-2/install/bin/extract_features ./modelsaver/last.caffemodel ./test_grl.prototxt txt_prob ./Feature/$((iter))/wiki_txt_prob 11 leveldb GPU ${gpu}

done

