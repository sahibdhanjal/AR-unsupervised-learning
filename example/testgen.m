clear, clc;
j=1;
scale = 40;
X=zeros(scale^3,3);
for x_point=1:1:scale
    for y_point=1:1:scale
        for z_point=1:1:scale
            X(j,:)=[x_point,y_point,z_point];
            j=j+1;
        end
    end
end

mu_1=[5, 8, 7];
mu_2=[12, 14, 18];
mu_3=[33, 29, 27];
sigma_1=3*eye(3);
sigma_2=3*eye(3);
sigma_3=5*eye(3);
pi_1=.2;
pi_2=.1;
pi_3=.7;
Obs_Weight=pi_1*mvnpdf(X,mu_1,sigma_1)+pi_2*mvnpdf(X,mu_2,sigma_2)+pi_3*mvnpdf(X,mu_3,sigma_3);

output = [X,Obs_Weight];
dlmwrite('data.txt',output,'delimiter','\t','precision',15)