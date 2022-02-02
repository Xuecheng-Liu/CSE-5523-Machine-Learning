%% reduce the dimension of the training data
clear;
load("train79.mat");
X_train = d79;
y_train = vertcat(zeros(1000,1),ones(1000,1));
coeff = pca(X_train);
X_train = X_train*coeff(:,1:400);

%% reduce the dimension of testing data
load("test79.mat");
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
coeff = pca(X_test);
X_test = X_test*coeff(:,1:400);

%% initialize the weights
theta = zeros(400,1);
alpha = 0.0000000001;

te50 = [];
te200 = [];
te400 = [];
te1000 = [];
te2000 = [];

cost50 = [];
cost200 = [];
cost400 = [];
cost1000 = [];
cost2000 = [];
%% get error with 50 trainning examples
X = vertcat(X_train(1:25,:),X_train(1001:1025,:));
y = vertcat(y_train(1:25,:),y_train(1001:1025,:));
for i = 1:1000
cost = sum((X*theta - y).^2)/2;
gradient = transpose(X)*(X*theta - y);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge
te50(length(te50)+1) = sum((X_test*theta-y_test).^2)/2;
cost50(length(cost50)+1) = cost;
end

%% get error with 200 training examples
X = vertcat(X_train(1:100,:),X_train(1001:1100,:));
y = vertcat(y_train(1:100,:),y_train(1001:1100,:));
for i = 1:1000
cost = sum((X*theta - y).^2)/2;
gradient = transpose(X)*(X*theta - y);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge 
 te200(length(te200)+1) = sum((X_test*theta-y_test).^2)/2;
 cost200(length(cost200)+1) = cost;
end

%% get test error with 400 training examples
X = vertcat(X_train(1:200,:),X_train(1001:1200,:));
y = vertcat(y_train(1:200,:),y_train(1001:1200,:));
for i = 1:1000
cost = sum((X*theta - y).^2)/2;
gradient = transpose(X)*(X*theta - y);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge 
 te400(length(te400)+1) = sum((X_test*theta-y_test).^2)/2;
 cost400(length(cost400)+1) = cost;
end
%% get test error with 1000 training examples
X = vertcat(X_train(1:500,:),X_train(1001:1500,:));
y = vertcat(y_train(1:500,:),y_train(1001:1500,:));
for i = 1:1000
cost = sum((X*theta - y).^2)/2;
gradient = transpose(X)*(X*theta - y);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge 
 te1000(length(te1000)+1) = sum((X_test*theta-y_test).^2)/2;
 cost1000(length(cost1000)+1) = cost;
end
%% get test error with 2000 training examples
X = vertcat(X_train(1:1000,:),X_train(1001:2000,:));
y = vertcat(y_train(1:1000,:),y_train(1001:2000,:));
for i = 1:1000
cost = sum((X*theta - y).^2)/2;
gradient = transpose(X)*(X*theta - y);
theta = theta - alpha*gradient; 
% increase learning rate to speed up cobverge 
 te2000(length(te2000)+1) = sum((X_test*theta-y_test).^2)/2;
 cost2000(length(cost2000)+1) = cost;
end
%% plot the test error v.s number of iterations for each training size
xa = linspace(1,1000,1000);
plot(xa,te50,xa,te200,xa,te400,xa,te1000,xa,te2000);
title('Problem 4: Test Error v.s # iteration');
xlabel("number of iterations");
ylabel('test error');
legend('50','200','400','1000','2000')
figure;

%% plot the cost after each training iteration as a reference for the convergence of gradient descent
plot(xa,cost50,xa,cost200,xa,cost400,xa,cost1000,xa,cost2000)
title("Problem 4: Cost v.s #iteration");
xlabel("number of iterations");
ylabel('cost');
legend('50','200','400','1000','2000')
%% Observations for Problem 4
% I plotted two figures, the first one is for Test Error v.s # iterations
% and the second one is loss v.s # iterations. As the second figure
% shows,the loss is decreasing after each iteration which means we are in
% the right track to train our model so that we can see how our model
% generalize to the testing data. From the first figure, there are several
% observations. First of all, the test error does not keep decreasing as
% the training goes on. Sometimes we have already reached the minimum loss
% with only several hundreds of iterations and the later training just
% overfitting the training data (cost for 400 training data shows this
% phenomenon). The general trend for other number of training examples
% seems okay since the cost generally decreasing as more training goes.
% However, the loss may blow up after more and more trainings and we can
% see this trend from the loss with 50 training examples. This may imply
% there is a point where we have the best generalized weights for testing
% data and after that point we are just overfitting the training data.
