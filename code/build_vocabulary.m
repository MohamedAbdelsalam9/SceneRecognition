function vocab = build_vocabulary(image_paths, vocab_size, num_categories, num_train_per_cat)

	start = tic;
	no_features = 100000; %## no of features for generating the kmeans
	features = zeros(128, no_features);
	no_features_each = uint32(no_features / (num_categories * num_train_per_cat)); %no of features per category per image to ensure that all categories are equally represented
	for i = 1:length(image_paths)
		I = single(mat2gray(imread(image_paths{i})));
		[f,d] = vl_dsift(I, 'STEP', 5, 'fast'); %## %generate SIFT features, with a step of 5 (search every five pixels) for faster generation
		order = randperm(size(d,2), no_features_each); %get random features from each image equal to the no. of features needed per image as specified above
		features(:,1+no_features_each*(i-1):no_features_each*i) = d(:,order); %put these samples of features in the features matrix to be used later in the Kmeans
	end
	features = single(features);
	vocab = vl_kmeans(features, vocab_size, 'Initialization', 'plusplus'); %apply kmeans plus plus which ensures that the initialization of the means is as far from each other as possible, so that to avoid as possible the drawbacks regarding the dependence of the kmeans onthe initialization
	telapsed = toc(start);
	fprintf('time for building vocabulary: %d secs\n', telapsed);

end
