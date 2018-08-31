% GMM Initialization Setup
rosshutdown; clc; clear; close all; clc;
delete *.mat;

global x_h y_h z_h pitch_h roll_h yaw_h;                % global variables for head
global x_1 y_1 z_1 pitch_1 roll_1 yaw_1;                % global variables for quad
global inp out master thresh num_iter verbose K flag;        % global variables for gmm

master = '192.168.1.3';                                 % IP of master
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
x_coord = [0:.1:4.0]; y_coord = [0:.1:4.0]; z_trash = [0:.1:2.5]; z_coord(1,1,:) = z_trash;
Q = zeros(length(y_coord),length(x_coord),length(z_coord));
x_mat = permute(repmat(x_coord',1,length(y_coord),length(z_coord)),[2 1 3]);
y_mat = permute(repmat(y_coord,length(x_coord),1,length(z_coord)),[2 1 3]);
z_mat = repmat(z_coord,length(x_coord),length(y_coord),1);
R = 10;
points = [x_mat(:),y_mat(:),z_mat(:)];
output = points;
save(inp,'output','-v6');

% Start GMM in background
if  exist(fullfile(cd, inp), 'file') == 2
    commandStr = ['python emgmm.py ' inp ' ' out ' ' int2str(num_iter) ' ' int2str(thresh) ' ' int2str(verbose) ' ' int2str(K) '& echo$!'];
    [status, cmdout] = system(commandStr);
end

% pause so that the initial GMM computation 
% can be done in the background and resulting
% mat files created
pause(1);

%% Main Loop
s = cputime;
e = 0;
while true
    % fprintf('head: (%.2f, %.2f, %.2f) | quad: (%.2f, %.2f, %.2f)\n',x_h, y_h, z_h, x_1, y_1, z_1);

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
    Q = Q+Si;
    output = resample([points,Q(:)]);
    save(inp,'output','-v6');

    % update means/covars/weights every 0.5 sec
    % as GMM takes 0.2-0.4 sec on average
    if e >= 0.5
        s = cputime;
        load(out)
        means
    end
    
    % Publishing Control Commands
    if x_1>xpos
        vx = -v_x;
    else
        vx = v_x;
    end

    if y_1>ypos
        vy = -v_y;
    else
        vy = v_y;
    end

    if z_1>zpos
        vz = -v_z;
    else
        vz = v_z;
    end

    Rot_Mat=[cos(pitch_1)*cos(yaw_1),(sin(roll_1)*sin(pitch_1)*cos(yaw_1)-cos(roll_1)*sin(yaw_1)),(cos(roll_1)*sin(pitch_1)*cos(yaw_1)+sin(roll_1)*sin(yaw_1)); ...
     cos(pitch_1)*sin(yaw_1),(sin(roll_1)*sin(pitch_1)*sin(yaw_1)+cos(roll_1)*cos(yaw_1)),(cos(roll_1)*sin(pitch_1)*sin(yaw_1)-sin(roll_1)*cos(yaw_1)); ...
     -sin(pitch_1),(sin(roll_1)*cos(pitch_1)),(cos(roll_1)*cos(pitch_1))];
    Control_body=Rot_Mat\[vx;vy;vz];

    msg.Translation.X = Control_body(1);
    msg.Translation.Y = Control_body(2);
    msg.Translation.Z = Control_body(3);
    msg.Rotation.X = roll;
    msg.Rotation.Y = pitch;
    msg.Rotation.Z = yaw;

    send(publisher,msg);
    
    e = cputime - s;

    drawnow
end

