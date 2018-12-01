%%objective find best K

close all;
clc ;
clear;

%initialization
A=load("synthetic.mat");
A2train=A.knnClassify2dTrain;
A2test=A.knnClassify2dTest;

K=1:1:100;
D2trainMat=zeros(100,100);
D2testMat=zeros(100,100);

ErrorMatTrain=zeros(100,1);
ErrorMatTest=zeros(100,1);

ErrorCount_train=0;
ErrorCount_test=0;

%distance between a row with a matrix
for i = 1:100
    D2train=pdist2(A2train(i,1:2),A2train(:,1:2));
    D2trainMat(i,1:100)=D2train;
    D2test=pdist2(A2test(i,1:2),A2train(:,1:2));
    D2testMat(i,1:100)=D2test;
end

for i = 1:100
    [MinKtrain,I_train]=mink(D2trainMat,K(i),2);
    [MinKtest,I_test]=mink(D2testMat,K(i),2);
    for k = 1:i
        for j = 1:100
            if I_train(j,k)<=50
                ClassTrain(j,k)=1;
            else
                ClassTrain(j,k)=2;
            end
            if I_test(j,k)<=50
                ClassTest(j,k)=1;
            else
                ClassTest(j,k)=2;
            end
        end
    end
    Mtrain=mode(ClassTrain,2);
    Mtest=mode(ClassTest,2);
    for z = 1:100
        if Mtrain(z)~=A2train(z,3)
            ErrorCount_train=ErrorCount_train+1;
        end
        if Mtest(z)~=A2test(z,3)
            ErrorCount_test=ErrorCount_test+1;
        end
    end
    ErrorMatTrain(i)=ErrorCount_train/100;
    ErrorMatTest(i)=ErrorCount_test/100;
    ErrorCount_train=0;
    ErrorCount_test=0;
end

figure()
plot(ErrorMatTrain,'b')
title("ErrorRate for each K")
xlabel("K")
ylabel("Error")
grid on 
hold on
plot(ErrorMatTest,'r')
legend("train","test")

