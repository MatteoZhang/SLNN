%%objective find best K

close all;
clc;
clear;

%initialization
A=load("speech_dataset.mat");
A2train=A.dataset(1:4323,1:6);
A2test=A.dataset(4324:end,1:6);

%set K
K=1:1:100;
point2train=size(A2train,1);
row2test=size(A2test,1);

D2trainMat=zeros(point2train,point2train);
D2testMat=zeros(row2test,point2train);

ErrorMatTrain=zeros(length(K),1);
ErrorMatTest=zeros(length(K),1);

ErrorCount_train=0;
ErrorCount_test=0;

%distance between a row with a matrix
for i = 1:point2train
    D2train=pdist2(A2train(i,1:5),A2train(:,1:5));
    D2trainMat(i,1:size(A2train,1))=D2train;
end
for i = 1:row2test
    D2test=pdist2(A2test(i,1:5),A2train(:,1:5));
    D2testMat(i,1:point2train)=D2test;
end
for i = 1:length(K)
    [MinKtrain,I_train]=mink(D2trainMat,K(i),2);
    [MinKtest,I_test]=mink(D2testMat,K(i),2);
    for k = 1:i
        for j = 1:point2train
            %A2train with index I
            if A.dataset(I_train(j,k),6)==1
                ClassTrain(j,k)=1;
            else
                ClassTrain(j,k)=2;
            end
        end
        for j = 1:row2test
            if A.dataset(I_test(j,k),6)==1
                ClassTest(j,k)=1;
            else
                ClassTest(j,k)=2;
            end
        end
    end
    
    %mode of the classes
    Mtrain=mode(ClassTrain,2);
    Mtest=mode(ClassTest,2);
    for z = 1:length(K)
        if Mtrain(z)~=A2train(z,6)
            ErrorCount_train=ErrorCount_train+1;
        end
        if Mtest(z)~=A2test(z,6)
            ErrorCount_test=ErrorCount_test+1;
        end
    end
    ErrorMatTrain(i)=ErrorCount_train/length(K);
    ErrorMatTest(i)=ErrorCount_test/length(K);
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