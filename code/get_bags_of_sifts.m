function image_feats = get_bags_of_sifts(image_paths, vocab)

	start = tic;
	vocab_size = size(vocab, 2);
	image_feats = zeros(length(image_paths),vocab_size); %image features as to be used in SVM later, the number of features is the vocabulary size
	for i = 1:length(image_paths)
		I = single(mat2gray(imread(image_paths{i})));
		[f,d] = vl_sift(I, 'PeakThresh', 0.01, 'EdgeThresh', 2, 'Levels', 4); %##extracting the SIFT describtor of the image, the Peak threshold and Edge threshold were specified after trials with different values
	    	d=single(d);
	    
	    	index = zeros(1,size(d,2));
	    	for n = 1:size(d,2) % iteration of col of d(each feature in the image)
			temp = 1000000;
			for j = 1:vocab_size %iteration of col of vocab and calculating the euclidean distance between each visual word and the feature, to assign it to the nearest word
			    	distance = sum((d(:,n) - vocab(:,j)) .^2);
				if (min(distance,temp) == distance)
					index(n) = j;
					temp = distance;
				end
				
			end
			image_feats(i,index(n)) = image_feats(i,index(n)) + 1; %add 1 to the word nearest to the feature (in the histogram of words of each image)
		end
	end
	telapsed = toc(start);
	fprintf('time for getting bags of words: %d secs\n', telapsed);

end
