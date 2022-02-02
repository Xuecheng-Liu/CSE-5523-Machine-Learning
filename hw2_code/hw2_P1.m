%% Problem 1
clc;
clear;
fprintf("----------Problem 1----------\n");
% load the data
load("train79.mat")
% prepare X_trian and y_train
X_train = d79;
y_train = vertcat(zeros(1000,1),ones(1000,1));
% fit the SVM model
SVMModel = fitcsvm(X_train,y_train);

% load test data
load("test79.mat");
X_test = d79;
% make predictions
yhat = predict(SVMModel,X_test);
y_test = y_train;
count = 0;
% calculate the accuracy
for i = 1:2000
    if yhat(i) == y_test(i)
        count = count + 1;
    end
end 
accuracy = (count /2000)*100;
fprintf("The accuracy with standard SVM is %f%%.\n",accuracy)

%% now use standard least linear classifier

clear;
% load the data
load("train79.mat")
% prepare X_trian and y_train
X_train = d79;
y_train = vertcat(zeros(1000,1),ones(1000,1));
% solve the weights directly with normal equation
W = pinv(transpose(X_train)*X_train)*transpose(X_train)*y_train;
% load the test data
load("test79.mat");
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
predict = X_test*W;
% yhat is the actualy prediction of labels
yhat = [];
for n = 1: length(predict)
    if predict(n) > 0.5
        yhat = [yhat;1];
    else
        yhat = [yhat;0];
    end 
end

count = 0;
for i = 1:2000
    if yhat(i) == y_test(i)
        count = count + 1;
    end
end 
accuracy = (count/2000)*100;
fprintf("The accuracy with standard least squared is %f%%.\n",accuracy)

