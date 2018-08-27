clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------- ROS Environment Setup -------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setenv('ROS_IP','localhost');
setenv('ROS_MASTER_URI', 'http://127.0.0.1:11311');
rosinit('NodeName', '/matlab');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------- GMM Parameters Setup -------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input = 'points.mat';
output = 'results.mat';
write = false;              % write data to .txt file or not
thresh = 6;     % means 1e-6
num_iter = 5;   % means 10^5
verbose = 1;    % set 1 if you want to print num iterations/ time
K = 5;          % expected number of gaussians

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------- ROS parameters - Astronet -----------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set up the domain
x_coord=[0:.1:4.0]; %meters
y_coord=[0:.1:4.0]; %meters
z_trash=[0:.1:2.5]; %meters
z_coord(1,1,:)=z_trash;
clear z_trash;

%Define Visual Interest Accumulation function, The INPUTS WEIGHTS for Sahib
Q=zeros(length(y_coord),length(x_coord),length(z_coord));

x_mat = permute(repmat(x_coord',1,length(y_coord),length(z_coord)),[2 1 3]);
y_mat = permute(repmat(y_coord,length(x_coord),1,length(z_coord)),[2 1 3]);
z_mat = repmat(z_coord,length(x_coord),length(y_coord),1);

R=10; %Range of Human Vision: doesn't really matter as it extends outside of frame

%Define the human head state x_h y_h z_h pitch_h roll_h yaw_h
x_h=3; y_h=2; z_h=1.8; pitch_h=0; roll_h=0; yaw_h=pi/2;

%simulate human walking in a circle
%This loop will be a while loop in the final implementation
for t=1:1:100
    %Human head measurements come in from VICON here. I use simulated data
    x_h=2+cos(t/10);
    y_h=2+sin(t/10);
    yaw_h=atan2(sin(t/10),cos(t/10))+pi/2;
    %keep the rest of the head states constant for now


    phi=real(acos((((x_mat-x_h).*cos(yaw_h).*cos(pitch_h)+(y_mat-y_h).*sin(yaw_h).*cos(pitch_h)-(z_mat-z_h).*sin(pitch_h))./sqrt((x_mat-x_h).^2+(y_mat-y_h).^2+(z_mat-z_h).^2)))); 
    %For now we have a constant c1; however, in the future will add a LIDAR
    %to the headset so that This can be switched out for very narrow Gaussian
    %With spike centered at appropriate distance from human

    %If the algorithm doesn't work last minute, just add in the physical locations of
    %walls and objects and project the Gaussiant point to here and say in the
    %text that this is a simulation of Lidar data which will be implemented in
    %future work
    c1=1;
    %c2 is the Field of View function which is a truncated Gaussian based on
    %the physical characteristics of the human macula
    a=-60; %deg
    b=60; %deg
    sigma=8; %deg
    c2=(1/sqrt(2*pi))*exp(-(1/2)*(phi*180./pi).^2./sigma^2)./(sigma.*((1/2)*(1+erf(b/(sqrt(2)*sigma)))-(1/2)*(1+erf(a/(sqrt(2)*sigma)))));
    B=(1/max(0,c1))+(1/max(0,c2));
    Si=1./B;
    Si=Si./max(max(max(Si)));

    %Now Update the Visual Interest Accumulation Function
    Q=Q+Si;
    save(fullfile(cd,input),'Q');

    % commandStr = ['python emgmm.py ' input ' ' output ' ' int2str(num_iter) ' ' int2str(thresh) ' ' int2str(verbose)];
    % system(commandStr);

end
%Plot The accumulated Q for fun
scatter3(x_mat(:),y_mat(:),z_mat(:),Q(:))