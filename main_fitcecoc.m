targetSize = [128,128];
location = fullfile('lfw');
parpool;
disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
tbl = countEachLabel(imds0);
mask = tbl{:,2}>=10 & tbl{:,2}<=80;
disp(['Number of images: ',num2str(sum(tbl{mask,2}))]);
persons = unique(tbl{mask,1});


[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

t=tiledlayout('flow');
nexttile(t);
montage(imds);

disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;

[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;

% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% NOTE: Rows of V are observations, columns are features.
singularValues = diag(S);
variance = singularValues.^2;
totalVariance = sum(variance);
cumulativeVariance = cumsum(variance) / totalVariance;
k = find(cumulativeVariance >= 0.95, 1);
k = min(size(V,2),k);

% Discard unnecessary data
W = S * V';                             % Transform V to weights (ala PCA)
W = W(1:k,:);                           % Keep first K weights
U = U(:,1:k);                           % Keep K eigenfaces

% Find feature vectors of all images
X = W';
Y = categorical(imds.Labels, persons);

% Create colormap
cm=[1,0,0;
    0,0,1,
    0,1,0];
% Assign colors to target values
c=cm(1+mod(uint8(Y),size(cm,1)),:);

disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;

% You may try this, to get a more optimized model
% 'OptimizeHyperparameters','all',...

Mdl = fitcecoc(X, Y,'Verbose', 2,'Learners','svm',...
               'Options',options);
toc;

%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

% Save the model and persons that the model recognizes.
% NOTE: An important part of the submission.
save('model','Mdl','persons','U', 'targetSize');
