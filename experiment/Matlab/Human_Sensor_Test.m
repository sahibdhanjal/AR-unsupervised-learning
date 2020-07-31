% GMM Initialization Setup
rosshutdown; clc; clear; close all; clc;


global x_h y_h z_h pitch_h roll_h yaw_h;                % global variables for head
global x_1 y_1 z_1 pitch_1 roll_1 yaw_1;                % global variables for quad
global inp out master thresh num_iter verbose K flag;        % global variables for gmm

master = '192.168.1.4';                                 % IP of master
inp = 'points.mat';                                     % name of the file to which the weighting matrix is written
out = 'results.mat';                                    % name of the file to which outputs are written
flag = 1;

thresh = 6;                                             % means 1e-6
num_iter = 5;                                           % means 10^5
verbose = 0;                                            % set 1 if you want to print num iterations/ time
K = 3;                                                  % expected number of gaussians

xpos = 0; ypos = 0; zpos = 1;                           % position to hover around
v_x = 0.05; v_y = 0.05; v_z = 0.2;                      % set velocities for each dimension
yaw = 0; pitch = 0; roll = 0;                           % set YPR for the quad

%% ROS Environment Setup
setenv('ROS_IP',master);
setenv('ROS_MASTER_URI', ['http://' master ':11311']);
rosinit('NodeName', '/matlab');

topic_list = {'/vicon/William/William' '/vicon/hbirddg/hbirddg'};
callback_list = {@viconCallback_Head @viconCallback_Quad};

headpose = rossubscriber(topic_list{1},'geometry_msgs/TransformStamped',callback_list{1});
quadpose = rossubscriber(topic_list{2},'geometry_msgs/TransformStamped',callback_list{2});

publisher = rospublisher('/controller','geometry_msgs/Transform');
msg = rosmessage(publisher);

%% Domain Setup for Control Loop
dx=.1;dy=.1;dz=.1;dpitch=.01;dyaw=.01;droll=.01;
x_coord = [0:.1:4.0]; y_coord = [0:.1:4.0]; z_trash = [0:.1:2.5]; z_coord(1,1,:) = z_trash;
Q = zeros(length(y_coord),length(x_coord),length(z_coord));
Si=zeros(length(y_coord),length(x_coord),length(z_coord));

%simulated Lidar
Lidar=zeros(length(y_coord),length(x_coord),length(z_coord));
Lidar(1:29,1:2,9:22)=ones(length(1:29),length(1:2),length(9:22)); %board
Lidar(24:41,33:41,9:11)=ones(length(24:41),length(33:41),length(9:11)); %buzzer Table
Lidar(34:41,21:41,9:11)=ones(length(34:41),length(21:41),length(9:11)); %assembly table
%Lidar(:,:,1:2)=ones(41,41,2);
S_R=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_pdx=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_mdx=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_pdy=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_mdy=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_pdz=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_mdz=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_pdpitch=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_mdptich=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_pdyaw=zeros(length(y_coord),length(x_coord),length(z_coord));
S_R_mdyaw=zeros(length(y_coord),length(x_coord),length(z_coord));
u_R_save=zeros(1,10^6);
v_R_save=zeros(1,10^6);
w_R_save=zeros(1,10^6);
s_R_save=zeros(1,10^6);
u_R_filt_save=zeros(1,10^6);
v_R_filt_save=zeros(1,10^6);
w_R_filt_save=zeros(1,10^6);
s_R_filt_save=zeros(1,10^6);
tune_save_u=zeros(1,10^6);
tune_save_v=zeros(1,10^6);
tune_save_w=zeros(1,10^6);
yaw_h_save=zeros(1,10^6);
dt_save=zeros(1,10^6);
psi_H_save=zeros(41,41,10^6);
psi_H=zeros(length(y_coord),length(x_coord),length(z_coord));
psi_Old=zeros(length(y_coord),length(x_coord),length(z_coord));
x_mat = permute(repmat(x_coord',1,length(y_coord),length(z_coord)),[2 1 3]);
y_mat = permute(repmat(y_coord,length(x_coord),1,length(z_coord)),[2 1 3]);
z_mat = repmat(z_coord,length(x_coord),length(y_coord),1);
R = 10; %Human, should be infinity so needs to change.
Ri=1.5; %Robot %9-10-18
Ri=5.5;
alphaa=60*pi/180;


%TEST
K_u=10^9;K_v=10^9;K_w=10^9;K_s=25^6;

xy_sat=.5;
z_sat=.5;
d_s=1;
kappa=.02;
points = [x_mat(:),y_mat(:),z_mat(:)];
output = points;
save(inp,'output');

% Start GMM in background
if  exist(fullfile(cd, inp), 'file') == 2 && flag == 1
    commandStr = ['python emgmm.py ' inp ' ' out ' ' int2str(num_iter) ' ' int2str(thresh) ' ' int2str(verbose) ' ' int2str(K) ' &'];
%     system(commandStr);
    flag = 0;
end

% pause so that the initial GMM computation
% can be done in the background and resulting
% mat files created
pause(1);

%% Main Loop
count=0;
s = 0;
e = 1;
tic;
dt=toc;
tic;
while true
    count=count+1;
    x_temp=x_1; y_temp=y_1; z_temp=z_1;
      if x_temp>max(x_coord)
        x_temp=max(x_coord);
    end
    if x_temp<min(x_coord)
        x_temp=min(x_coord);
    end
    
    if y_temp>max(y_coord)
        y_temp=max(y_coord);
    end
    if y_temp<min(y_coord)
        y_temp=min(y_coord);
    end
    
    if z_temp>max(z_coord)
        z_temp=max(z_coord);
    end
    if z_temp<min(z_coord)
        z_temp=min(z_coord);
    end  
    % fprintf('head: (%.2f, %.2f, %.2f) | quad: (%.2f, %.2f, %.2f)\n',x_h, y_h, z_h, x_temp, y_temp, z_temp);
    % Control Script
    phi = real(acos((((x_mat-x_h).*cos(yaw_h).*cos(pitch_h)+(y_mat-y_h).*sin(yaw_h).*cos(pitch_h)-(z_mat-z_h).*sin(pitch_h))./sqrt((x_mat-x_h).^2+(y_mat-y_h).^2+(z_mat-z_h).^2))));
    c1 = 1;
    a = -60; %deg
    b = 60; %deg
    sigma = 8; %deg

    c2 = (1/sqrt(2*pi))*exp(-(1/2)*(phi*180./pi).^2./sigma^2)./(sigma.*((1/2)*(1+erf(b/(sqrt(2)*sigma)))-(1/2)*(1+erf(a/(sqrt(2)*sigma)))));
    B = (1/max(0,c1))+(1/max(0,c2));
    Si = 1./B;
    Si = Si./max(max(max(Si)));
   
    %%%%%Check this later to make sure it works in practice.
    S_barrier=(Si<10^-10);
    %%%%
    Q = Q+Si.*Lidar;
    output = resample([points,Q(:)]);
    save(inp,'output','-v6');



    % update means/covars/weights every 0.5 sec
    % as GMM takes 0.2-0.4 sec on average
    if e >= 0.5
        s = cputime;
        try
            load(out)
            means
        catch
            warning('Not able to load .mat file');
            means
        end
    end
    
    e = cputime - s;
    psi_Old=psi_H;
    %psi_H_temp=weights(1)*mvnpdf(points,means(1,:),diag(covars(1,:)))+weights(2)*mvnpdf(points,means(2,:),diag(covars(2,:)))+weights(3)*mvnpdf(points,means(3,:),diag(covars(3,:)));
    psi_H_temp=weights(1)*mvnpdf(points,means(1,:),eye(3))+weights(2)*mvnpdf(points,means(2,:),eye(3))+weights(3)*mvnpdf(points,means(3,:),eye(3));
    psi_H=reshape(psi_H_temp,[41,41,26]);
    psi_H=psi_H./sum(psi_H(:));
    psi_H_save(:,:,count)=psi_H(:,:,10);
    %mesh(psi_H(:,:,13))
    psi_H=psi_H.*S_barrier;
    %%%%%%%%PLUS DX%%%%%%%%%%%
   x_i=x_temp+dx;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_temp).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_temp).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_temp).^2....
    +(z_mat-z_temp).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_temp).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_temp).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_pdx=1./B;
S_R_pdx=S_R_pdx./sum(S_R_pdx(:));
x_i=x_i-dx;
%%%%%%MINUSDX%%%%%%%
x_i=x_i-dx;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_temp).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_temp).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_temp).^2....
    +(z_mat-z_temp).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_temp).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_temp).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_mdx=1./B;
S_R_mdx=S_R_mdx./sum(S_R_mdx(:));
x_i=x_i+dx;
%%%%%%PLUSDY%%%%%%%
y_i=y_temp+dy;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_temp).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_temp).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_temp).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_pdy=1./B;
S_R_pdy=S_R_pdy./sum(S_R_pdy(:));
y_i=y_i-dy;
%%%%%%MINUSDY%%%%%%%
y_i=y_i-dy;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_temp).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_temp).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_temp).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_mdy=1./B;
S_R_mdy=S_R_mdy./sum(S_R_mdy(:));
y_i=y_i+dy;
%%%%%%PLUSDZ%%%%%%%
z_i=z_temp+dz;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_i).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_pdz=1./B;
S_R_pdz=S_R_pdz./sum(S_R_pdz(:));;
z_i=z_i-dz;
%%%%%%MINUSDZ%%%%%%%
z_i=z_i-dz;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_i).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_mdz=1./B;
S_R_mdz=S_R_mdz./sum(S_R_mdz(:));;
z_i=z_i+dz;
%%%%%%PLUSDPITCH%%%%%%%
pitch_i=pitch_1+dpitch;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_i)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_i)-...
    (z_mat-z_i).*sin(pitch_i))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_pdpitch=1./B;
S_R_pdpitch=S_R_pdpitch./sum(S_R_pdpitch(:));;
pitch_i=pitch_i-dpitch;
%%%%%%MINUSDpitch%%%%%%%
pitch_i=pitch_i-dpitch;

phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_i)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_i)-...
    (z_mat-z_i).*sin(pitch_i))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_mdpitch=1./B;
S_R_mdpitch=S_R_mdpitch./sum(S_R_mdpitch(:));;
pitch_i=pitch_i+dpitch;
%%%%%%PLUSDYaw%%%%%%%
yaw_i=yaw_1+dyaw;

phi...
    =acos((((x_mat-x_i).*cos(yaw_i).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_i).*cos(pitch_1)-...
    (z_mat-z_i).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_pdyaw=1./B;
S_R_pdyaw=S_R_pdyaw./sum(S_R_pdyaw(:));;
yaw_i=yaw_i-dyaw;
%%%%%%MINUSDYaw%%%%%%%
yaw_i=yaw_i-dyaw;

phi...
    =acos((((x_mat-x_i).*cos(yaw_i).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_i).*cos(pitch_1)-...
    (z_mat-z_i).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R_mdyaw=1./B;
S_R_mdyaw=S_R_mdyaw./sum(S_R_mdyaw(:));;
yaw_i=yaw_i+dyaw;

%%Nominal
phi...
    =acos((((x_mat-x_i).*cos(yaw_1).*cos(pitch_1)...
    +(y_mat-y_i).*sin(yaw_1).*cos(pitch_1)-...
    (z_mat-z_i).*sin(pitch_1))...
    ./sqrt((x_mat-x_i).^2....
    +(y_mat-y_i).^2....
    +(z_mat-z_i).^2)));

c1=real(Ri^2-(x_mat-x_i).^2-(y_mat-y_i).^2-(repmat(z_coord,length(x_coord),length(y_coord),1)-z_i).^2);
c2=real(alphaa-phi);
c3=real(alphaa+phi);
B=(1/max(0,c1))+(1/max(0,c2))+(1/max(0,c3));
S_R=1./B;
S_R=S_R./sum(S_R(:));;

S_R_dx=(S_R_pdx-S_R_mdx)./(2*dx);
S_R_dy=(S_R_pdy-S_R_mdy)./(2*dy);
S_R_dz=(S_R_pdz-S_R_mdz)./(2*dz);
S_R_dpitch=(S_R_pdpitch-S_R_mdpitch)./(2*dpitch);
S_R_dyaw=(S_R_pdyaw-S_R_mdyaw)./(2*dyaw);

S_R(isnan(S_R))=0;
S_R_dx(isnan(S_R_dx))=0;
S_R_dy(isnan(S_R_dy))=0;
S_R_dz(isnan(S_R_dz))=0;
S_R_dyaw(isnan(S_R_dyaw))=0;
S_R_dpitch(isnan(S_R_dpitch))=0;



a0=trapz(trapz(trapz(2.*max(0,S_R-psi_H).*(psi_H-psi_Old)./dt,1),2),3).*dx.*dy.*dz;
a1=trapz(trapz(trapz(2.*max(0,S_R-psi_H).*(S_R_dx.*cos(pitch_1).*cos(yaw_1)+S_R_dy.*cos(pitch_1).*sin(yaw_1)+S_R_dz.*(-sin(pitch_1))),1),2),3).*dx.*dy.*dz;
a2=trapz(trapz(trapz(2.*max(0,S_R-psi_H).*(S_R_dx.*(sin(roll_1).*sin(pitch_1).*cos(yaw_1)-cos(roll_1).*sin(yaw_1))+S_R_dy.*(sin(roll_1).*sin(pitch_1).*sin(yaw_1)+cos(roll_1).*cos(yaw_1))+S_R_dz.*(sin(roll_1).*cos(pitch_1))),1),2),3).*dx.*dy.*dz;
a3=trapz(trapz(trapz(2.*max(0,S_R-psi_H).*(S_R_dx.*(cos(roll_1).*sin(pitch_1).*cos(yaw_1)+sin(roll_1).*sin(yaw_1))+S_R_dy.*(cos(roll_1).*sin(pitch_1).*sin(yaw_1)-sin(roll_1).*cos(yaw_1))+S_R_dz.*(cos(roll_1).*cos(pitch_1))),1),2),3).*dx.*dy.*dz;
a6=trapz(trapz(trapz(2.*max(0,S_R-psi_H).*(S_R_dyaw.*(cos(roll_1).*sec(pitch_1))+S_R_dpitch.*(-sin(roll_1))),1),2),3).*dx.*dy.*dz;

u_bar=-K_u*(a1-a0*a1^(-1)*(K_u+K_v+K_w+K_s)^(-1));
v_bar=-K_v*(a2-a0*a2^(-1)*(K_u+K_v+K_w+K_s)^(-1));
w_R=-K_w*(a3-a0*a3^(-1)*(K_u+K_v+K_w+K_s)^(-1));

tune_save_u(count)=u_bar;
tune_save_v(count)=v_bar;
tune_save_w(count)=w_R;
if sqrt(u_bar^2+v_bar^2+w_R^2)>.3
    fprintf('Saturated\n')
%This Works Satur
temp_norm=sqrt(u_bar^2+v_bar^2+w_R^2);
u_bar=.3*u_bar/temp_norm;
v_bar=.3*v_bar/temp_norm;
w_R=.3*w_R/temp_norm;
end

s_R=-K_s*(a6-a0*a6^(-1)*(K_u+K_v+K_w+K_s)^(-1));
if isnan(u_bar)==1
    u_bar=0;
end
if isnan(v_bar)==1
    v_bar=0;
end
if isnan(w_R)==1
    w_R=0;
end
if isnan(s_R)==1
    s_R=0;
end

Rot_Mat=[cos(pitch_1)*cos(yaw_1),(sin(roll_1)*sin(pitch_1)*cos(yaw_1)-cos(roll_1)*sin(yaw_1)),(cos(roll_1)*sin(pitch_1)*cos(yaw_1)+sin(roll_1)*sin(yaw_1)); ...
     cos(pitch_1)*sin(yaw_1),(sin(roll_1)*sin(pitch_1)*sin(yaw_1)+cos(roll_1)*cos(yaw_1)),(cos(roll_1)*sin(pitch_1)*sin(yaw_1)-sin(roll_1)*cos(yaw_1)); ...
     -sin(pitch_1),(sin(roll_1)*cos(pitch_1)),(cos(roll_1)*cos(pitch_1))];

%abs added. May need to remove later to make consistent with theory
rho_l=kappa*abs((norm([x_temp;y_temp]-[x_h;y_h])^2-d_s*norm([x_temp;y_temp]-[x_h;y_h]))^(-2))*inv(Rot_Mat)*([x_temp;y_temp;0]-[x_h;y_h;0]);
rho_1=kappa*abs((norm([x_temp;y_temp]-[means(1,1);means(1,2)])^2-d_s*norm([x_temp;y_temp]-[means(1,1);means(1,2)]))^(-2))*inv(Rot_Mat)*([x_temp;y_temp;0]-[means(1,1);means(1,2);0]);
rho_2=kappa*abs((norm([x_temp;y_temp]-[means(2,1);means(2,2)])^2-d_s*norm([x_temp;y_temp]-[means(2,1);means(2,2)]))^(-2))*inv(Rot_Mat)*([x_temp;y_temp;0]-[means(2,1);means(2,2);0]);
rho_3=kappa*abs((norm([x_temp;y_temp]-[means(3,1);means(3,2)])^2-d_s*norm([x_temp;y_temp]-[means(3,1);means(3,2)]))^(-2))*inv(Rot_Mat)*([x_temp;y_temp;0]-[means(3,1);means(3,2);0]);
(rho_l+rho_1+rho_2+rho_3)'


u_R=u_bar+rho_l(1)+rho_1(1)+rho_2(1)+rho_3(1);
v_R=v_bar+rho_l(2)+rho_1(2)+rho_2(2)+rho_3(2);
bound_vx=[0;0;0];
bound_vy=[0;0;0];
bound_vz=[0;0;0];
bound_vc=[0;0;0];
bound_flag=0;
    U_max=.2;
    if x_temp==max(x_coord)
        bound_vx=Rot_Mat\[-U_max;0;0];
        bound_flag=1;
    end
    if x_temp==min(x_coord)
        bound_vx=Rot_Mat\[U_max;0;0];
        bound_flag=1;
    end

    if y_temp==max(y_coord)
        bound_vy=Rot_Mat\[0;-U_max;0];
        bound_flag=1;
    end
    if y_temp==min(y_coord)
        bound_vy=Rot_Mat\[0;U_max;0];
        bound_flag=1;
    end

    if z_temp==max(z_coord)
        bound_vz=Rot_Mat\[0;0;-U_max];
        bound_flag=1;
    end
    if z_temp==min(z_coord)
        bound_vz=Rot_Mat\[0;0;U_max];
        bound_flag=1;
    end
   if bound_flag==1
       bound_vc=bound_vx+bound_vy+bound_vz;
       u_R=bound_vc(1);
       v_R=bound_vc(2);
       w_R=bound_vc(3);
   end
    u_R_save(count)=u_R;
    v_R_save(count)=v_R;
    w_R_save(count)=w_R;
    s_R_save(count)=s_R;
    yaw_h_save(count)=yaw_h;



    msg.Translation.X = u_R;
    msg.Translation.Y = v_R;
    msg.Translation.Z = w_R;
    msg.Rotation.Z = s_R;
    %if count>10
     %msg.Translation.X=median(u_R_save(count-3:count));
     %msg.Translation.Y=median(v_R_save(count-3:count));
     %msg.Translation.Z=median(w_R_save(count-3:count));
     %msg.Translation.X=median(u_R_save(count-3:count));
     %msg.Rotation.Z =median(s_R_save(count-3:count));  
     %u_R_filt_save(count)=median(u_R_save(count-3:count));
     %v_R_filt_save(count)=median(v_R_save(count-3:count));
     %w_R_filt_save(count)=median(w_R_save(count-3:count));
     %s_R_filt_save(count)=median(s_R_save(count-3:count));
    %end
    
    msg.Rotation.X = 0;
    msg.Rotation.Y = 0;


    send(publisher,msg);
    dt=toc;
    tic;
    dt_save(count)=dt;
    drawnow
end
