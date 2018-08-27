%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian Mixture Model Data Generation
% Author - Sahib Dhanjal <dhanjalsahib@gmail.com>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = generator(mu, sigma, scale, pi, matfile, write)
    j=1;
    X = zeros(scale^3,3);
    for x_point=1:1:scale
        for y_point=1:1:scale
            for z_point=1:1:scale
                X(j,:)=[x_point,y_point,z_point];
                j=j+1;
            end
        end
    end
    Obs_Weight=pi(1)*mvnpdf(X,mu(1,:),sigma(1)*eye(3))+pi(2)*mvnpdf(X,mu(2,:),sigma(2)*eye(3))+pi(3)*mvnpdf(X,mu(3,:),sigma(3)*eye(3));
    output = resample([X,Obs_Weight]);
    
    % save data to .mat file for python
    save(fullfile(cd,matfile),'output');
    
    % write to file if filename given
    if write==1
        filename = strcat(matfile(1:end-4),'.txt');
        dlmwrite(filename, output, 'delimiter', '\t', 'precision', 15);
    end
end