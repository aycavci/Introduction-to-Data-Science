%% Loading data
data = csvread('featuresFlowCapAnalysis.csv');
labels = csvread('labelsFlowCapAnalysis.csv');
traindata = data(1:179,:);
testdata = data(180:end,:);
datamean = mean(traindata);
[normalizedtrain, average, sigma] = zscore(traindata);
normalizedtest = (testdata-average)/sigma;

trainlabels = [];
for i = 1:179
    trainlabels = [trainlabels, "Train"];
end
testlabels = [];
for i = 1:179
    testlabels = [testlabels, "Test"];
end
labelscpy = string(labels);
labelscpy(labelscpy == "1") = "Healthy";
labelscpy(labelscpy == "2") = "AML";

%% PCA
[coeff,score,latent,tsquared,explained,mu] = pca(traindata);
cumulativesum = cumsum(explained);

% plotting variance explained
figure(1)
stairs(cumulativesum(1:80))
ylim([0 100])
xlim([0 80])
ylabel('Variance explained [%]')
xlabel('Number of principal components')

% Using pca to reduce to 2 dimensions. 
% # of 2 labels = 23, # of 1 labels = 156
figure(2)
reduceddata = traindata*coeff(:,1:2);
x = reduceddata(:,1);
y = reduceddata(:,2);
% first 6 eigenvalues explain 90.26 \% variance
gscatter(x,y,labelscpy)
xlabel('PC1')
ylabel('PC2')



%% ANOVA for feature selection
fvalues = [];
for i = 1:length(normalizedtrain(:,1))
    feature = traindata(:,i);
    f = myOneWayANOVA(feature,labels);
    fvalues = [fvalues, [f;i]];
end

% highest f values are most explaining features
sortedfvalues = sortrows(fvalues', 'descend');
% boxplots of 3 most explaining features

for i = 1:3
    feature = traindata(:, sortedfvalues(i,2));
    figure(i+2)
    boxplot(feature, labelscpy)
    title(strcat("Feature ", string(sortedfvalues(i,2))))
end

% boxplots of 3 least explaining features
len = length(labelscpy);
for i = 1:3
    feature = traindata(:, sortedfvalues(len-i+1,2));
    figure(i+5)
    boxplot(feature, labelscpy)
    title(strcat("Feature ", string(sortedfvalues(len-i+1,2))))
end

%% TrainvsTest distribution
totaldifferences = 0;
for i = 1:186
    datalabelled = traindata(:,i);
    dataunlabelled = testdata(:,i);
    [p,h] = ranksum(datalabelled, dataunlabelled);
    totaldifferences = totaldifferences + h;
    if h == 1
        boxplot([traindata(:,i); testdata(:,i)], ...
        [trainlabels,testlabels])
    end
end
totaldifferences

%% tsne for feature reduction
% spearman distance seems to do quite well, others really divide the class
% 1 data in 2 clusters
% normalized data seems to work quite well as well
rng('default')

figure(1)
Y = tsne(traindata,'Algorithm','exact');
gscatter(Y(:,1),Y(:,2),labelscpy)
title("Train data")

figure(2)
Y = tsne(normalizedtrain,'Algorithm','exact');
gscatter(Y(:,1),Y(:,2),labelscpy)
title("Normalized train data")

figure(3)
numberofcomponents = 20;
reduceddata = traindata*coeff(:,1:numberofcomponents);
Y = tsne(reduceddata,'Algorithm','exact');
gscatter(Y(:,1),Y(:,2),labelscpy)
title("PCA reduced train data")

figure(4)
numberoffeatures = 20;
newdataset = [];
for i = 1:numberoffeatures
    feature = traindata(:, sortedfvalues(i,2));
    newdataset = [newdataset,feature];
end
Y = tsne(newdataset,'Algorithm','exact');
gscatter(Y(:,1),Y(:,2),labelscpy)
title("ANOVA reduced train data")
