%% forked from Paolo-26

close all; clc; clear;
data = load('heightWeight.mat');

F = 40;
M = 25;

male = data.heightWeightData(data.heightWeightData(:,1) == 1,2:end);
female = data.heightWeightData(data.heightWeightData(:,1) == 2,2:end);

test_M = male(1:M,:);
test_F = female(1:F,:);
train_M = male(M+1:end,:);
train_F = female(F+1:end,:);

% MLE mean (males).
mMales = 0;
for i = 1:length(train_M)
    mMales = mMales + train_M(i,:);
end
mMales = mMales/length(train_M);

% MLE mean (females).
mFemales = 0;
for i = 1:length(train_F)
    mFemales = mFemales + train_F(i,:);
end
mFemales = mFemales/length(train_F);

% MLE covariance (males).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(male)
    firstTerm = firstTerm + male(i,:)'*male(i,:); 
end
firstTerm = firstTerm/length(male); 
secondTerm = mMales.*mMales';
sigmaMales = firstTerm - secondTerm; % covariance matrix

% MLE covariance (females).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(female)
    firstTerm = firstTerm + female(i,:)'*female(i,:); 
end
firstTerm = firstTerm/length(female);
secondTerm = mFemales.*mFemales';
sigmaFemales = firstTerm - secondTerm

pie(1) = 0.5;
pie(2) = 0.5;

for i = 1:length(train_M(:,1))
     num = pie(1)*norm(2*pi*sigmaMales)^(1/2)*exp((train_M(i,:)...
         -mMales)*inv(sigmaMales)*(train_M(i,:)-mMales)') 
     den1 = pie(1)*norm(2*pi*sigmaMales)^(1/2)*exp((train_M(i,:)...
         -mMales)*inv(sigmaMales)*(train_M(i,:)-mMales)')
     den2 = pie(1)*norm(2*pi*sigmaFemales)^(1/2)*exp((train_F(i,:)...
         -mFemales)*inv(sigmaFemales)*(train_F(i,:)-mFemales)')
     PosteriorM_train(i) = num/(den1+den2)
end