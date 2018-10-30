%%objective find best K

close all;
clc ;
clear;

A=load("synthetic.mat");
A2train=A.knnClassify2dTrain;
A2test=A.knnClassify2dTest;
K=1:1:100;
D2trainMat=zeros(100,100);

for i = 1:100
    D2train=pdist2(A2train(i,1:2),A2train(:,1:2));
    D2trainMat(i,1:100)=D2train;
end
for i = 1:100
    MinKelement=mink(D2train,K(i),2);
end

% you can also write like this
% for i = 1:100
%     for j = 1:100
%         Dtrain(i,j)= sqrt(abs((A2train(i,1)-A2train(j,1))^2-(A2train(i,2)
%                      -A2train(j,2))^2));
%     end
% end