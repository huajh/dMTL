%We recomend to pre-process the data using either of the following two methods:
% from Deng Cai
% http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html

%Nomalize each vector to unit
%===========================================
function fea = pre_proc_feature(fea)

    [nSmp,nFea] = size(fea);
    for i = 1:nSmp
         fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
    end
    %===========================================

    %Scale the features (pixel values) to [0,1]
    %===========================================
    maxValue = max(max(fea));
    fea = fea/maxValue;