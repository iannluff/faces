function YPred = recognize_faces(RGB)
% RECOGNIZE_FACES - map images of faces to people names    
%  YPred = recognize_faces(RGB) accepts a cell array RGB of images, which
% should be RGB images. YPred returns a categorical array of image
% labels.
    ;
    % Load precomputed model from a MAT file. For a format
    % of the file, see the M-file main_fitcecoc.m
    load('model.mat');
    num_images = size(RGB, 3);
    Grayscale = cell(1, num_images);
    parpool;

    % Get grayscale images of the desired size
    parfor i = 1:num_images
        Grayscale{i} = imresize(im2gray(RGB{i}), targetSize);
    end

    %Grayscale = cellfun(@(I)imresize(im2gray(I), targetSize),...
    %                   RGB, 'uni',false);

    B = cat(3,Grayscale{:});
    D = prod(targetSize);
    B = reshape(B, D, []);

    % Normalizing data...';
    B = single(B)./256;
    [B, C, SD] = normalize(B);

    % Extract weights
    W = U' * B;

    % Predict faces
    X = W';
    
    YPred = predict(Mdl, X);
end


