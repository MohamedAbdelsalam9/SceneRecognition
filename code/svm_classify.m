%train a one vs all linear SVM classifier, and apply each classifier of the 15 to each image
%and the category is assigned based on the highest result
%5 fold cross validation is applied to get the best regularization parameter (lambda) and avoid overfitting

function predicted_categories = svm_classify(train_features, labels, test_features, categories)

	start = tic;
	lambda = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 10]; % Regularization parameters to choose from
	accuracy_f = zeros(1,length(lambda)); %accuracy when using each lambda in the cross validation step
	maxIter = 100000; % Maximum number of iterations
	fold = 5.; %leave five items from the data set each iteration for cross validation
	for i = 1 : length(categories)	
		for j = 1:length(labels)
			if strcmp(labels(j),categories(i)) y(j) = 1; %%convert the training labels to either 1 (for this category) or -1 (for all other categories) to train one vs all classifier for each category
			else	y(j) = -1;
			end
		end

		%take a sample from the negatives to prevent it's over representation (1:3 positives to negatives), with the negatives taken randomly from all the negatives
		no_positives = sum(y==1);
		order_ = randperm(length(labels),no_positives*4);
		order = ones(1,no_positives*3);
		count = 1;
		%make sure that this random sample contains negatives only (as positives will be added in the next step)
		for k = 1:length(order_)
			if count > length(order)
				break;
			end
			if (order_(k) < (i-1)*no_positives + 1 || order_(k) > i*no_positives) order(count) = order_(k); count = count + 1;
			end
		end

		%put all the positives in the beginning for easier manipulation
		[y_1 I] = sort(y,'descend'); y = [y_1(1:no_positives) y_1(order)]; 
		x = [train_features(:,I(1:no_positives)) train_features(:,order)];
		no_folds = uint16(length(y)/fold); %%the no of iterations needed to span the whole training set based on the fold size
	
		%cross validation for regularization parameter
		for k = 1:length(lambda)
			accuracy = zeros(1,no_folds);
			for j = 1 : no_folds
				x_ = [x(:,1:(j-1)*fold) x(:, j*fold+1:size(x,2))];
				y_ = [y(1:(j-1)*fold) y(j*fold+1:size(x,2))];
				[w_ b_] = vl_svmtrain(x_, y_, lambda(k), 'MaxNumIterations', maxIter);
				accuracy(j) = sum(y((j-1)*fold+1:j*fold) == sign(w_'*x(:,(j-1)*fold+1:j*fold) + b_)) / fold;
			end
			accuracy_f(k) = sum(accuracy) / double(no_folds); %%cross validation accuracy of each lambda
		end
		[a(i) ind(i)] = max(accuracy_f); %%get the best lambda
		[w(:,i) b(i)] = vl_svmtrain(x, y, lambda(ind(i)), 'MaxNumIterations', maxIter); %%the training on the whole data set with the lambda choosen
	end

	%classifying the test set
	for i = 1 : length(test_features)	
		classification = w'*test_features(:,i) + b';
		[acc(i) indx(i)] = max(classification);
		predicted_categories(i,1) = categories(indx(i));
	end

	telapsed = toc(start);
	fprintf('time for getting classifying: %d secs\n', telapsed)	

end
