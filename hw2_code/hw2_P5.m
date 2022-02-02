%% Problem 5

% load the training data
clc;
clear;
fprintf("----------Problem 5----------\n");

load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));

% split the training data
X_train = X(1:1600,:);
X_vali = X(1601:2000,:);
Y_train = Y(1:1600,:);
Y_vali = Y(1601:2000,:);
% array for possible bandwidths 5 - 100
bw = [1,200,400,600,800,1000];
%%
performance = zeros(1,6); %% accuracy for each bandwidth
for t = 1:6
    bandwidth = bw(t);
    KernelMatrix = ConstructKernel(X_train,bandwidth); %construct kernel matrix
    alpha = pinv(KernelMatrix)*Y_train;
    weightedSum = zeros(400,1);
    for i = 1:400
        score = 0;
        % score for a test data
        for j = 1:1600
            score = score + alpha(j)*GaussianKernel(X_train(j,:),X_vali(i,:),bandwidth); % modify this
        end
        weightedSum(i) = score;
    end
performance(t) = accuracy(weightedSum,Y_vali);
end
plot(bw,performance)
xlabel("bandwidth");
ylabel("accuracy on validation set");
title("bandwidth v.s accuracy")
%% helper functions
% compute the kernel matric
function KernelMatrix = ConstructKernel(X_train,bandwidth)
    KernelMatrix = zeros(length(X_train),length(X_train));
    for i = 1:length(X_train)
        for j = 1:length(X_train)
            KernelMatrix(i,j) = GaussianKernel(X_train(i,:),X_train(j,:),bandwidth);
        end
    end
end

% compute each entry in kernel matrix
function entry = GaussianKernel(xi,xj,bandwith)
    entry = exp(-((xi - xj)*transpose(xi - xj))/(2*bandwith^2));
end

function acc = accuracy(weightedSum,Y_vali)
    pre = zeros(400,1);
    for c = 1: 400
        if weightedSum(c)>0.5
            pre(c) = 1;
        end
    end
    count = 0;
    for c = 1:400
        if pre(c) == Y_vali
            count = count + 1;
        end
    end
    acc = count/400;
end

