%% ----------Initialization------------


load('Label.mat'); %Load labels of test set. Make sure Label.mat matches the label of your test set!

te_n_I = size(teCatAll,1);
te_n_T = size(teCatAll,1);
teImgCat = teCatAll;
teTxtCat = teCatAll;

%% -------------------Search Task Definition(Wikipedia)-----------------------
I_te = importdata('wiki_img_prob_te/feature.txt'); %Load Img common representation
T_te = importdata('wiki_txt_prob_te/feature.txt'); %Load Txt common representation
I_tr = I_tr(1:2173, :);
T_tr = T_tr(1:2173, :);
I_te = I_te(1:693, :);
T_te = T_te(1:693, :);


disp('Test');
D = pdist2([I_te; T_te], 'cosine');
Z = -D;

% %Image->Text
QryonTestBi(Z, teImgCat, teTxtCat);
% %Text->Image
QryonTestBi(Z', teTxtCat, teImgCat);


