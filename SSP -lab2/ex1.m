%% forked from Paolo-26

close all; clc; clear;
data = load('heightWeight.mat');

male = data.heightWeightData(data.heightWeightData(:,1) == 1,2:end);
female = data.heightWeightData(data.heightWeightData(:,1) == 2,2:end);

%% Scatter plot (males and females)
figure(1)
hold on
scatter(male(:,1),male(:,2),'b')
scatter(female(:,1),female(:,2),'r')
grid on
xlabel('Height (cm)')
ylabel('Weight (kg)')
legend('Males','Females')
xlim([120 220])
ylim([30 130])

%% Histogram (males)
figure(2)
hold on
edges=120:5:220;
h1 = histcounts(male(:,1),edges);
h2 = histcounts(female(:,1),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
xlabel('Height (cm)')
ylabel('Number of people')
grid on
legend('Males','Females')
 
%% Histogram (females)
figure(3)
edges=30:5:130;
h1 = histcounts(male(:,2),edges);
h2 = histcounts(female(:,2),edges);
b = bar(edges(1:end-1),[h1;h2]',1);
b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
xlabel('Weight (kg)')
ylabel('Number of people')
grid on
legend('Male','Female')

%% MLE mean (males).
mMales = 0;
for i = 1:length(male)
    mMales = mMales + male(i,:);
end
mMales = mMales/length(male)

%% MLE mean (females).
mFemales = 0;
for i = 1:length(female)
    mFemales = mFemales + female(i,:);
end
mFemales = mFemales/length(female)

%% MLE covariance (males).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(male)
    firstTerm = firstTerm + male(i,:)'*male(i,:); 
end
firstTerm = firstTerm/length(male); 
secondTerm = mMales.*mMales';
sigmaMales = firstTerm - secondTerm % covariance matrix

%% MLE covariance (females).
firstTerm = zeros(2);  % 2x2 matrix of zeros
for i = 1:length(female)
    firstTerm = firstTerm + female(i,:)'*female(i,:); 
end
firstTerm = firstTerm/length(female);
secondTerm = mFemales.*mFemales';
sigmaFemales = firstTerm - secondTerm

%% Multivariate gaussian plot (males)
figure(4)
x1 = 120:1:220; x2 = 30:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mMales,sigmaMales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - males');
title('Males')
view(0,90);colorbar;

%% Multivariate gaussian plot (females)
figure(5)
x1 = 120:1:220; x2 = 30:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mFemales,sigmaFemales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density - females');
title('Females')
view(0,90); colorbar;

%% Multivariate gaussian plot (males)
figure(7)
x1 = 120:1:220; x2 = 30:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mMales,sigmaMales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
view(0,90);colorbar;

%% Multivariate gaussian plot (females)
hold on
x1 = 120:1:220; x2 = 30:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mFemales,sigmaFemales);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([120 220 30 130 0 max(F(:))])
xlabel('Height (cm)'); ylabel('Weight (kg)'); zlabel('Probability Density');
title('Multivariate of the joint Gaussian')
view(0,90); colorbar;


