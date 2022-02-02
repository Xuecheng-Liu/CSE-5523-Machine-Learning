% Notice: please run the code section by section
%% Problem 8 K mean clustering (K = 2)
clc; clear;
load('train79.mat')
X = d79;
[idx,C] = kmeans(X,2);
keySet = [-1,-1]; % cluser number
valueSet = [0,1]; % class label
% find the index of point in each class
id1 = [];
id2 = [];
for i = 1:2000
    if idx(i) == 1
        id1(length(id1)+1) = i;
    else
        id2(length(id2)+1) = i;
    end
end
% find majority of each class
c7 = 0;
c9 = 0;
for i = 1:length(id1)
    if id1(i) < 1001
        c7 = c7 + 1;
    elseif id1(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    keySet(1) = 1;
else
    keySet(2) = 2;
end
c7 = 0;
c9 = 0;
for i = 1:length(id2)
    if id2(i) < 1001
        c7 = c7+1;
    elseif id2(i) > 1000
        c9 = c9 +1;
    end
end
if c7 > c9
    keySet(1) = 1;
else
    keySet(2) = 2;
end
% map for cluster and label
M = containers.Map(keySet,valueSet);

% load the test data
load('test79.mat');
X_test = d79;
%assign cluster to each test point
predict = zeros(2000,1);
for i = 1:2000
    D1 = norm(X_test(i,:)- C(1,:));
    D2 = norm(X_test(i,:)- C(2,:));
    if D1 < D2
        predict(i) = 1;
    else
        predict(i) = 2;
    end
end
% get predicted label
y_hat = zeros(2000,1);
for i = 1:2000
    y_hat(i) = M(predict(i));
end
y_test = vertcat(zeros(1000,1),ones(1000,1));
acc = 0;
for i = 1:2000
    if y_hat(i) == y_test(i)
        acc = acc + 1;
    end
end
fprintf("The accuracy when K = 2 is %f%%",100*acc/2000);
%% K means with K = 5
clc; clear;
load('train79.mat')
X = d79;
[idx,C] = kmeans(X,5);
keySet = [1,2,3,4,5];
valueSet = [-1,-1,-1,-1,-1]; % label for each cluster
M = containers.Map(keySet,valueSet);
id1 = [];
id2 = [];
id3 = [];
id4 = [];
id5 = [];   
for i = 1:2000
    if idx(i) == 1
        id1(length(id1)+1) = i;
    elseif idx(i) ==2
        id2(length(id2)+1) = i;
    elseif idx(i) == 3
        id3(length(id3)+1) = i;
    elseif idx(i) == 4
        id4(length(id4)+1) = i;
    else
        id5(length(id5)+1) = i;
    end
end
% find majority for first cluster
c7 = 0;
c9 = 0;
for i = 1:length(id1)
    if id1(i) < 1001
        c7 = c7 + 1;
    elseif id1(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(1) = 0;
else
    M(1) = 1;
end

% find majority for second cluster
c7 = 0;
c9 = 0;
for i = 1:length(id2)
    if id2(i) < 1001
        c7 = c7 + 1;
    elseif id2(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(2) = 0;
else
    M(2) = 1;
end
% find majority for thrid cluster
c7 = 0;
c9 = 0;
for i = 1:length(id3)
    if id3(i) < 1001
        c7 = c7 + 1;
    elseif id3(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(3) = 0;
else
    M(3) = 1;
end

% find majority for 4th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id4)
    if id4(i) < 1001
        c7 = c7 + 1;
    elseif id4(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(4) = 0;
else
    M(4) = 1;
end

% find majority for 5th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id5)
    if id5(i) < 1001
        c7 = c7 + 1;
    elseif id5(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(5) = 0;
else
    M(5) = 1;
end

load('test79.mat')
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
predict = zeros(2000,1);
for i = 1:2000
    dis = zeros(1,5);
    for j = 1:5
        dis(j) = norm(X_test(i,:)-C(j,:));
    end
    [val,id] = min(dis);
    predict(i) = id;
end
y_hat = zeros(2000,1);
for i = 1:2000
    y_hat(i) = M(predict(i));
end
acc = 0;
for i = 1:2000
    if y_hat(i) == y_test(i)
        acc = acc + 1;
    end
end
fprintf("The accuracy when K = 5 is %f%%",100*acc/2000);
%% K = 10
clc; clear;
load('train79.mat')
X = d79;
[idx,C] = kmeans(X,10);
keySet = [1,2,3,4,5,6,7,8,9,10];
valueSet = zeros(1,10);
M = containers.Map(keySet,valueSet);
id1 = [];
id2 = [];
id3 = [];
id4 = [];
id5 = [];  
id6 = [];
id7 = [];
id8 = [];
id9 = [];
id10 = [];
for i = 1:2000
    if idx(i) == 1
        id1(length(id1)+1) = i;
    elseif idx(i) ==2
        id2(length(id2)+1) = i;
    elseif idx(i) == 3
        id3(length(id3)+1) = i;
    elseif idx(i) == 4
        id4(length(id4)+1) = i;
    elseif idx(i) == 5
        id5(length(id5)+1) = i;
    elseif idx(i) == 6
        id6(length(id6)+1) = i;
    elseif idx(i) == 7
        id7(length(id7)+1) = i;
    elseif idx(i) == 8
        id8(length(id8)+1) = i;
    elseif idx(i) == 9
        id9(length(id9)+1) = i;
    elseif idx(i) == 10
        id10(length(id10)+1) = i;
    end
end
% find majority for first cluster
c7 = 0;
c9 = 0;
for i = 1:length(id1)
    if id1(i) < 1001
        c7 = c7 + 1;
    elseif id1(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(1) = 0;
else
    M(1) = 1;
end

% find majority for second cluster
c7 = 0;
c9 = 0;
for i = 1:length(id2)
    if id2(i) < 1001
        c7 = c7 + 1;
    elseif id2(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(2) = 0;
else
    M(2) = 1;
end
% find majority for thrid cluster
c7 = 0;
c9 = 0;
for i = 1:length(id3)
    if id3(i) < 1001
        c7 = c7 + 1;
    elseif id3(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(3) = 0;
else
    M(3) = 1;
end

% find majority for 4th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id4)
    if id4(i) < 1001
        c7 = c7 + 1;
    elseif id4(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(4) = 0;
else
    M(4) = 1;
end

% find majority for 5th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id5)
    if id5(i) < 1001
        c7 = c7 + 1;
    elseif id5(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(5) = 0;
else
    M(5) = 1;
end

% find majority for first cluster
c7 = 0;
c9 = 0;
for i = 1:length(id6)
    if id6(i) < 1001
        c7 = c7 + 1;
    elseif id6(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(6) = 0;
else
    M(6) = 1;
end

% find majority for second cluster
c7 = 0;
c9 = 0;
for i = 1:length(id7)
    if id7(i) < 1001
        c7 = c7 + 1;
    elseif id7(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(7) = 0;
else
    M(7) = 1;
end
% find majority for thrid cluster
c7 = 0;
c9 = 0;
for i = 1:length(id8)
    if id8(i) < 1001
        c7 = c7 + 1;
    elseif id8(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(8) = 0;
else
    M(8) = 1;
end

% find majority for 4th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id9)
    if id9(i) < 1001
        c7 = c7 + 1;
    elseif id9(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(9) = 0;
else
    M(9) = 1;
end

% find majority for 5th cluster
c7 = 0;
c9 = 0;
for i = 1:length(id10)
    if id10(i) < 1001
        c7 = c7 + 1;
    elseif id10(i) > 1000
        c9 = c9 + 1;
    end
end
if c7 > c9
    M(10) = 0;
else
    M(10) = 1;
end

load('test79.mat')
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
predict = zeros(2000,1);
for i = 1:2000
    dis = zeros(1,10);
    for j = 1:10
        dis(j) = norm(X_test(i,:)-C(j,:));
    end
    [val,id] = min(dis);
    predict(i) = id;
end
y_hat = zeros(2000,1);
for i = 1:2000
    y_hat(i) = M(predict(i));
end
acc = 0;
for i = 1:2000
    if y_hat(i) == y_test(i)
        acc = acc + 1;
    end
end
fprintf("The accuracy when K = 10 is %f%%",100*acc/2000);
%% K = 50
clc; clear;
load('train79.mat')
X = d79;
[idx,C] = kmeans(X,50);

l1 = zeros(1,50); % count of point in all clusters

for i = 1:50
    for j = 1:2000
        if idx(j) == i
            l1(i) = l1(i)+1;
        end
    end
end

l2 = zeros(1,50); % number of sevens in each cluster
for i = 1:50
    for j = 1:1000
        if idx(j) == i
            l2(i) = l2(i)+1;
        end
    end
end

keySet = linspace(1,50,50); % each entry represents a cluster
valueSet = ones(1,50); % label for each cluster
for i = 1:50
    if l1(i) - l2(i) < l2(i)
        valueSet(i) = 0;
    end
end

M = containers.Map(keySet,valueSet);

load('test79.mat');
X_test = d79;
y_test = vertcat(zeros(1000,1),ones(1000,1));
predict = zeros(2000,1);
for i = 1:2000
    dis = zeros(1,50);
    for j = 1:50
        dis(j) = norm(X_test(i,:)-C(j,:));
    end
    [val,id] = min(dis);
    predict(i) = id;
end

y_hat = zeros(2000,1);
for i = 1:2000
    y_hat(i) = M(predict(i));
end
acc = 0;
for i = 1:2000
    if y_hat(i) == y_test(i)
        acc = acc + 1;
    end
end
fprintf("The accuracy when K = 50 is %f%%",100*acc/2000);
