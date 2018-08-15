%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample Importance Resampling Algorithm
% Author - Sahib Dhanjal <dhanjalsahib@gmail.com>
% Read more here - http://cecas.clemson.edu/~ahoover/ece854/lecture-notes/lecture-pf.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = resample(mat)
    [N, D] = size(mat);
    points = mat(:, 1:D-1);
    weights = mat(:, D);
    weights = weights./sum(weights);
    output = zeros(N,D-1);
    
    % basic resampling preparation
    Q = cumsum(weights);
    t = rand(N+1, 1);
    T = sort(t);
    T(N+1) = 1;
    
    % resampling index generation
    i = 1; j = 1;
    idxs = zeros(N,1);
    while i<=N
        if T(i)<Q(j)
            idxs(i) = j;
            i = i+1;
        else
            j = j+1;
        end
    end
    
    % put points in matrix
    for i=1:N
        output(i,:) = points(idxs(i),:);
    end
end