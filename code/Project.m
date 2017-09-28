% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
run('vlfeat/toolbox/vl_setup')

data_path = '../data/';

%the list of the categories
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use.
num_train_per_cat = 100; %##
num_categories = length(categories);

%% we used this function from James Hayes Course to load the image paths
%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);        
        
% build the vocabulary (visual words) that will be used as features later using build_vocabulary.m, it will get the SIFT describtor of these
%images first, then use kmeans with k specified as the vocabulary size
%If the code has been run before and there is an existing vocabulary (vocab.mat), the code doesn't regenerate vocabulary
%so to regenerate vocabulary, delete vocab.mat in the directory
if ~exist('vocab.mat', 'file')
fprintf('No existing visual word vocabular y found. Computing one from training images\n')
vocab_size = 400; %Larger values will work better (to a point) but be slower to compute
vocab = build_vocabulary(train_image_paths, vocab_size, num_categories, num_train_per_cat);
save('vocab.mat', 'vocab')
else
vocab = importdata('vocab.mat');
end	
        
% get the features for each image to be used later in SIFT using get_bags_of_sifts.m
%If the code has been run before and there is an existing svm_features (svm_train_features.mat, svm_test_features.mat), the code doesn't regenerate these features
%so to regenerate features, delete both files in the directory
if ~(exist('svm_train_features.mat', 'file') && exist('svm_test_features.mat', 'file'))
svm_train_features = get_bags_of_sifts(train_image_paths, vocab);
svm_test_features  = get_bags_of_sifts(test_image_paths, vocab);
save('svm_train_features.mat', 'svm_train_features')
save('svm_test_features.mat', 'svm_test_features')
else
svm_train_features = importdata('svm_train_features.mat');
svm_test_features = importdata('svm_test_features.mat');
end

%train the SVM from the training features done in the previous step (which is a histogram of visual words for each image), then apply the trained weights on the test set and put the labels in the predicted_categories vector to compare it later with the test labels and get the accuracy
predicted_categories = svm_classify(svm_train_features', train_labels', svm_test_features', categories);

%get the accuracy of each category
match = strcmp(predicted_categories,test_labels);
accuracy = zeros(length(categories),1);
for i = 1 : length(categories)
	accuracy(i) = sum(match((i-1)*num_train_per_cat + 1 : i*num_train_per_cat)) / num_train_per_cat;
	fprintf('accuracy of category(%d) is %.2f\n', i, accuracy(i));
end

%% we used this function from James Hayes course to visualize the data
% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called.
create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
