%% cross validation to find the best min leaf size for decision tree
clc; clear;
load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(X,Y,'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

%% use the best hyper parameter over the validation and test
clc;
clear;
load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));
t= fitctree(X,Y,'MinLeafSize',10);
load('test79.mat');
X_test = d79;
y_hat = predict(t,X_test);
y = Y;
count = 0;
for i = 1:2000
    if y(i) == y_hat(i)
        count = count + 1;
    end
end
fprintf("The accuracy with min leaf size = 10 over the test set is %f%%",count*100/2000);
%% cross validation in bagged tree to find the best number of tree
clc; clear;
load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));
% split the training data
X_train = X(1:1600,:);
X_vali = X(1601:2000,:);
Y_train = Y(1:1600,:);
Y_vali = Y(1601:2000,:);


bags = linspace(10,50,5);
N = numel(bags);
err = zeros(N,1);
for n = 1:N
   t = TreeBagger(bags(n),X_train,Y_train);
   y_hat = predict(t,X_vali);
   y_hat = str2double(y_hat);
   count = 0;
   for j = 1:length(y_hat)
       if Y_vali(j) == y_hat(j)
        count = count + 1;
        end
   end
   err(n) = 1 - (count/length(y_hat));
end

plot(bags,err);
xlabel('number of trees');
ylabel('cross-validated error');

%% Performance on test set for bagged trees with number of tree = 50
clc;
clear;
load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));
t= TreeBagger(50,X,Y);
load('test79.mat');
X_test = d79;
y_hat = predict(t,X_test);
y_hat = str2double(y_hat);
y = Y;
count = 0;
for i = 1:2000
    if y(i) == y_hat(i)
        count = count + 1;
    end
end
fprintf("The accuracy with number of trees = 50 over the test set is %f%%",count*100/2000);
%% Cross validation over boosted trees
clc; clear;
load("train79.mat")
X = d79;
Y = vertcat(zeros(1000,1),ones(1000,1));
% split the training data
X_train = X(1:1600,:);
X_vali = X(1601:2000,:);
Y_train = Y(1:1600,:);
Y_vali = Y(1601:2000,:);

err = zeros(3,1);
t1 = fitcensemble(X_train,Y_train,'Method','AdaBoostM1');
y_hat1 = predict(t1,X_vali);
count = 0;
for i = 1:400
    if y_hat1(i) == Y_vali(i)
        count = count +1;
    end
end
err(1) = 1 - (count/400);

t2 = fitcensemble(X_train,Y_train,'Method','RUSBoost');
y_hat2 = predict(t2,X_vali);
count = 0;
for i = 1:400
    if y_hat2(i) == Y_vali(i)
        count = count +1;
    end
end
err(2) = 1 - (count/400);

t3 = fitcensemble(X_train,Y_train,'Method','LogitBoost');
y_hat3 = predict(t3,X_vali);
count = 0;
for i = 1:400
    if y_hat3(i) == Y_vali(i)
        count = count +1;
    end
end
err(3) = 1 - (count/400);
plot([1,2,3],err);
xlabel('type of boosting');
ylabel('cross-validated error');
%% Test over Adaboosted decision trees
clc;
clear;
load("train79.mat")
X_train = d79;
Y_train = vertcat(zeros(1000,1),ones(1000,1));
t = fitcensemble(X_train,Y_train,'Method','AdaBoostM1');
load('test79.mat');
X_test = d79;
y_hat = predict(t,X_test);
y = Y_train;
count = 0;
for i = 1:2000
    if y(i) == y_hat(i)
        count = count + 1;
    end
end
fprintf("The accuracy with Adaboost over the test set is %f%%",count*100/2000);