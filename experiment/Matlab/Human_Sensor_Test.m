clc
rosshutdown
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------- Sim Environment Setup -------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global x_h y_h z_h pitch_h roll_h yaw_h;    % global variables for head
global x_1 y_1 z_1 pitch_1 roll_1 yaw_1;     % global variables for quad

flag = true;                                % using ROS or not
master = '127.0.0.1';                       % IP of master

inp = 'points.mat';                         % name of the file to which the weighting matrix is written
out = 'results.mat';                        % name of the file to which outputs are written
write = false;                              % write data to .txt file or not
thresh = 6;                                 % means 1e-6
num_iter = 5;                               % means 10^5
verbose = 0;                                % set 1 if you want to print num iterations/ time
K = 3;                                      % expected number of gaussians
time = 10000;                                 % cycles for which the main loop should run

orig_x = 0; orig_y = 0; orig_z = 0;         % set origin in each dimension
v_x = 0.05; v_y = 0.05; v_z = 0.2;          % set velocities for each dimension
yaw = 0; pitch = 0; roll = 0;               % set YPR for the quad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------- ROS Callback Service -------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (flag)
    % set ROS Environment
    setenv('ROS_IP',master);
    setenv('ROS_MASTER_URI', ['http://' master ':11311']);
    rosinit('NodeName', '/matlab');  

    % topics to track and callback function
    topic_list = {'/vicon/William/William' '/vicon/hbirddg/hbirddg'};
    callback_list = {@viconCallback_Head @viconCallback_Quad};

    % subscribe to operator head and quadposition
    vicon_sub(1) = rossubscriber(topic_list{1},'geometry_msgs/TransformStamped',callback_list{1});
    vicon_sub(2) = rossubscriber(topic_list{2},'geometry_msgs/TransformStamped',callback_list{2});
    
    % Define loop rate
    rate = rosrate(60);

    % publish control messages to the quad
    ctrl_pub(1) = rospublisher('/controller','geometry_msgs/Transform');
    ctrl_msg(1) = rosmessage(ctrl_pub(1));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------- Main Loop for Simulation -----------------------%
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
t = 1;

points=[x_mat(:),y_mat(:),z_mat(:)];

%simulate human walking in a circle
%This loop will be a while loop in the final implementation
while 1
    %Human head measurements come in from VICON here. I use simulated data
%     x_h=2+cos(t/10);
%     y_h=2+sin(t/10);
%     yaw_h=atan2(sin(t/10),cos(t/10))+pi/2;
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
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %------------------------ GMM CUDA done here -------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    output = resample([points,Q(:)]);
    
    save(inp,'output','-v6');
    if  exist(fullfile(cd, inp), 'file') == 2
        commandStr = ['python emgmm.py ' inp ' ' out ' ' int2str(num_iter) ' ' int2str(thresh) ' ' int2str(verbose) ' ' int2str(K)];
        system(commandStr);
        
        if  exist(fullfile(cd, out), 'file') == 2
            load(out);
            % disp('Calculated Means | Covariances | Weights');
            % means
            % covars
            % weights
            t = t+1;
        end        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %---------------------- Quad control done here -----------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % control published only if ROS is working
    if (flag)     
        xpos = 0; ypos = 0; zpos = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% P Controllers for Position %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        

        % This will be converted to type asctec/mav_ctrl in 
        % a different node and published as rosgenmsg isn't 
        % able to generate messages for the quad
        Rot_Mat=[cos(pitch_1)*cos(yaw_1),(sin(roll_1)*sin(pitch_1)*cos(yaw_1)-cos(roll_1)*sin(yaw_1)),(cos(roll_1)*sin(pitch_1)*cos(yaw_1)+sin(roll_1)*sin(yaw_1)); ...
         cos(pitch_1)*sin(yaw_1),(sin(roll_1)*sin(pitch_1)*sin(yaw_1)+cos(roll_1)*cos(yaw_1)),(cos(roll_1)*sin(pitch_1)*sin(yaw_1)-sin(roll_1)*cos(yaw_1)); ...
         -sin(pitch_1),(sin(roll_1)*cos(pitch_1)),(cos(roll_1)*cos(pitch_1))];
        Control_body=Rot_Mat\[vx;vy;vz];
        
        ctrl_msg(1).Translation.X = Control_body(1);
        ctrl_msg(1).Translation.Y = Control_body(2);
        ctrl_msg(1).Translation.Z = Control_body(3);
       
        ctrl_msg(1).Rotation.X = roll;
        ctrl_msg(1).Rotation.Y = pitch;
        ctrl_msg(1).Rotation.Z = yaw;
        
        send(ctrl_pub(1),ctrl_msg(1));
    end
   drawnow 
end


%Plot The accumulated Q for fun
% scatter3(x_mat(:),y_mat(:),z_mat(:),Q(:))