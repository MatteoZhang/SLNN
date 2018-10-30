clear; close all; clc;
%%initializations
%mode per le classe dominante
%pdist2(b,A) per tutte le distanze tra b e ogni riga di A
A=load("synthetic.mat");
A2train=A.knnClassify2dTrain;
A2test=A.knnClassify2dTest;
k=3; % choose a k
K=1:1:100;

%%distace matrix
for i = 1:100
    for j = 1:100
        Dtrain(i,j)= sqrt(abs((A2train(i,1)-A2train(j,1))^2-(A2train(i,2)-A2train(j,2))^2));
    end
end
%%min distaces for k elements
MinKtrain=mink(Dtrain,k+1,2); 

for i = 1:100
    for j = 1:100
        for z = 2:k+1
            if MinKtrain(i,z) == Dtrain(i,j)
                MinKtrain(i,z+k) = j;
                if MinKtrain(i,z+k) < 51
                    MinKtrain(i,z+2*k)=1;
                else
                    MinKtrain(i,z+2*k)=2;
                end
            end
        end
    end
end

%%distace matrix
for i = 1:100
    for j = 1:100
        Dtest(i,j)= sqrt(abs((A2test(i,1)-A2train(j,1))^2-(A2test(i,2)-A2train(j,2))^2));
    end
end

%%min distaces
MinKtest=mink(Dtest,k+1,2);

for i = 1:100
    for j = 1:100
        for z = 2:k+1
            if MinKtest(i,z) == Dtest(i,j)
                MinKtest(i,z+k) = j;
                if MinKtest(i,z+k) < 51
                    MinKtest(i,z+2*k)=1;
                else
                    MinKtest(i,z+2*k)=2;
                end
            end
        end
    end
end
count = 0;
max = 0;
for i = 1:100
    for z = 2*k+2:3*k+1
        if MinKtrain(i,z) == 1
            count = count +1;
        else MinKtrain(i,z) == 2;
            max = max +1;
        end
    end
    if max < count
        MinKtrain(i,z+1)=1;
    else
        MinKtrain(i,z+1)=2;
    end
    max = 0;
    count = 0;
end
%% mode function
for i = 1:100
    for z = 2*k+2:3*k+1
        if MinKtest(i,z) == 1
            count = count +1;
        else MinKtest(i,z) == 2
            max = max +1;
        end
    end
    if max < count
        MinKtest(i,z+1)=1;
    else
        MinKtest(i,z+1)=2;
    end
    max = 0;
    count = 0;
end

figure
scatter(A2train(1:50,1),A2train(1:50,2),'r'); %red for class 1
hold on
scatter(A2train(51:100,1),A2train(51:100,2),'b'); %blue for class 2
hold on

figure
scatter(A2test(1:50,1),A2test(1:50,2),'r'); % test set 
hold on
scatter(A2test(51:100,1),A2test(51:100,2),'b'); 
hold on

             



