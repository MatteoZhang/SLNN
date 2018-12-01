clc; clear all;
data = load('Indian_Pines_Dataset');
indian_pines = data.indian_pines;
indian_pines_gt = data.indian_pines_gt;
C1 = 237;  % Corn class 4
C2 = 1265;  % Woods class 14
N_SPECTR = 220;
K = 200; %K<=220

n=0;
class1 = zeros(C1, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 4 % class index
            n = n + 1;
            class1(n,:) = indian_pines(i,j,:);
        end
    end
end

n = 0;
class2 = zeros(C2, N_SPECTR);
for i = 1:size(indian_pines, 1)
    for j = 1:size(indian_pines, 2)
        if indian_pines_gt(i,j)== 14 % class index
            n = n + 1;
            class2(n,:) = indian_pines(i,j,:);
        end
    end
end

class1 = class1-mean(class1);
class2 = class2-mean(class2);
classUnion = [class1 ; class2];
covUnion = cov(classUnion);
myCovUnion = zeros(N_SPECTR,N_SPECTR);
for i = 1:length(classUnion)
    tmp = classUnion(i,:)'*classUnion(i,:);
    myCovUnion = myCovUnion + tmp;
end
myCovUnion = myCovUnion/length(classUnion);
