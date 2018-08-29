%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Generation Parameters
clear, close all, clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set data parameters
mu = [ 1, 2, 1; 12, 14, 18; 6, 8, 7];
cov = [1, 2, 1];
scale = 20;
pi = [0.2 0.2 0.6];

% extra parameters for GMM, file I/O
write = false;  % write data to .txt file or not
thresh = 6;     % means 1e-6
num_iter = 5;   % means 10^5
verbose = 1;    % set 1 if you want to print num iterations/ time
K = 3;          % expected number of gaussians

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File Name for Input and Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set filenames of input and output .mat files
input = 'points.mat';
output = 'results.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Test Data
data = generator(mu, cov, scale, pi, input, write);
commandStr = ['python emgmm.py ' input ' ' output ' ' int2str(num_iter) ' ' int2str(thresh) ' ' int2str(verbose) ' ' int2str(K)];
system(commandStr);

if exist(output,'file')==2
    load(output);
    means
    covars
    weights
else
    disp('executing python in bg');
end
