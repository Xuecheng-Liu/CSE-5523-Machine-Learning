%% Problem 7 part 1
clc;
clear;
load ('train79.mat');
X_train = d79;
mean_matrix=X_train-repmat(mean(X_train,1),size(X_train,1),1);
[W, EvalueMatrix] = eig(cov(mean_matrix));
Evalues = diag(EvalueMatrix);
[Evalues,sidx] = sort(Evalues,'descend');
W = W(:,sidx);
W2 = W(:,1:2);
reduced = mean_matrix*W2; % 2-D data for 2000 training data
scatter(reduced(1:1000,1),reduced(1:1000,2),'.');
hold on
scatter(reduced(1001:2000,1),reduced(1001:2000,2),'.');
legend(['7';'9'],'location','southeast');

%% Problem 7 part 2

%% produce eigendigits for "7"
clc; clear;
load ('train79.mat');
X = d79(1:1000,:);
coeff = pca(X); %loadings for '7' class
colormap(gray);
x = reshape(coeff(:,600),28,28); % change this line for different loadings
y = x(:,28:-1:1);
pcolor(y)
%% eigendigits for "9"
clc; clear;
load ('train79.mat');
X = d79(1001:2000,:);
coeff = pca(X); %loadings for 9
colormap(gray);
x = reshape(coeff(:,5),28,28); % change this line for different loadings
y = x(:,28:-1:1);
pcolor(y)
%% eigendigits for both class
clc; clear;
load ('train79.mat');
X = d79;
coeff = pca(X); %loadings for all data
colormap(gray);
x = reshape(coeff(:,500),28,28); % change this line for different loadings
y = x(:,28:-1:1);
pcolor(y)
