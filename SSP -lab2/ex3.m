%see slide #82 -> D= dataset x is a row ,h=y cathegory
%forked from Paolo-26
clc; clear; close all;
data = load('XwindowsDocData.mat');
err = 0.05;

%in bag of words a = number of repetition and N numero delle parole

train = [sum(data.ytrain == 1) sum(data.ytrain == 2)];
test = [sum(data.ytest == 1) sum(data.ytest == 2)];
theta(:,1) = sum(data.xtrain(1:train(1),:) == 1)/train(1);
theta(:,2) = sum(data.xtrain(train(2)+1:end,:) == 1)/train(2);
pie(1) = train(1)/length(data.ytrain); %numero di valori presenti / tot
pie(2) = train(2)/length(data.ytrain);
uninformativeWords = (abs(theta(1:end,1)-theta(1:end,2))) <= err;
%MAP estimate
