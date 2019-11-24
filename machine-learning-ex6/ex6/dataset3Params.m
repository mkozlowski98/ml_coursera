function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C = values(1);
sigma = values(1);
error = 10000;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i=1:length(values)
  for j=1:length(values)
    C_current = values(i);
    sigma_current = values(j);
    model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
    predictions = svmPredict(model, Xval);
    actual_error = mean(double(predictions ~= yval));
    if actual_error < error
      error = actual_error;
      C = C_current;
      sigma = sigma_current;
    end
  end
end

% =========================================================================

end
