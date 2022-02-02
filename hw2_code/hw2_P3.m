%% Problem 3
% reduce the dimension of the training data
clear;
load("train79.mat");
X_train = d79;
y_train = vertcat(zeros(1000,1),ones(1000,1));
coeff = pca(X_train);
X_train = X_train*coeff(:,1:400);

% reduce the dimension of testing data
load("test79.mat");
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
coeff = pca(X_test);
X_test = X_test*coeff(:,1:400);

% prepare the arrary for plotting
num_train = linspace(50,2000,40);
error_SVM = [];
error_norm = [];
% iterate with 50, 100, ...2000 trainning examples
for i = 25:25:1000
    % choose the train data
    X = vertcat(X_train(1:i,:),X_train(1001:1000+i,:)); % choose half from both labels
    y = vertcat(y_train(1:i,:),y_train(1001:1000+i,:));
    W = pinv(transpose(X)*X)*transpose(X)*y; % get the weight directly
    MD = fitcsvm(X,y,'KernelScale','auto','BoxConstraint',100); % get weights by fitcsvm with C = 100
    
    
    % predictions with hard SVM model
    yhat_SVM = predict(MD,X_test);
    
    % predictions with normal equations
    prediction = X_test*W;
    yhat_norm = [];
    for n = 1:2000
        if prediction(n)>0.5
            yhat_norm = [yhat_norm;1];
        else
            yhat_norm = [yhat_norm;0];
        end
    end
    
    % computer the test error for both methods
    count_SVM = 0; % number of misclassifications
    count_norm = 0;
    % updata count of misclassifications
    for id = 1 : 2000
        if yhat_SVM(id) ~= y_test(id)
            count_SVM = count_SVM + 1;
        end
        if yhat_norm(id) ~= y_test(id)
            count_norm = count_norm + 1;
        end
    end
    error_SVM(length(error_SVM)+1) = count_SVM/2000;
    error_norm(length(error_norm)+1) = count_norm/2000;
    
end

% plot the test error for both methods
plot(num_train,error_SVM,'o',num_train,error_norm,'x');
title('Problem 3');
xlabel('number of training data');
ylabel('number of averaged test error');
legend('hard SVM','linear regression');

fprintf('Observations for Problem 3:\n');
fprintf('The result shows reducing the dimension of training data lowers the accuracy of predictions.\n');
fprintf('The result shows hard SVM has a higher test error than explict solutions.\n');
fprintf('The reason for this dues to the overfitting of training data');
fprintf('when we force C to be large which reduces the geometric margin.\n');
fprintf('The result also shows test error deceases in general as the model is trained with more data.\n');