%------------------------ Define viconCallback -----------------------%
function viconCallback_Head(src, msg)
    global x_h y_h z_h pitch_h roll_h yaw_h;
    x_h = msg.Transform.Translation.X+2;
    y_h = msg.Transform.Translation.Y+2;
    z_h = msg.Transform.Translation.Z-.1; %shift from top of head to eyes
    
    qx = msg.Transform.Rotation.X;
    qy = msg.Transform.Rotation.Y;
    qz = msg.Transform.Rotation.Z;
    qw = msg.Transform.Rotation.W;
    ans = quat2eul([qw qx qy qz]);
    
    % default is ZYX -> YPR 
    yaw_h = ans(1);
    pitch_h = ans(2);
    roll_h = ans(3); 
end
        
%---------------------------- END ------------------------------------%