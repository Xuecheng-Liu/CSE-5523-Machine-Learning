%% Problem 2 Implement Least Squares classifiers
fprintf("----------Problem 2----------\n")
clear;
% prepare training data
load("train79.mat");
X_train = d79;
y_train = vertcat(zeros(1000,1),ones(1000,1));

% prepare testing data
load("test79.mat");
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));

% initialize the weights
theta = zeros(784,1);

alpha = 0.0000000001;
for i = 1:1000
%cost = sum((X_train*theta - y_train).^2)/2;
gradient = transpose(X_train)*(X_train*theta - y_train);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge
if i > 800
    alpha = 0.00000000035;
end
end 

predict = X_test*theta;
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
fprintf("The accuracy with gradient descent least squared is %f%%.\n",accuracy)
fprintf("The result shows gradient descent result in a better performance\n")
