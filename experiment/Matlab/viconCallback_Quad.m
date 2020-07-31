%------------------------ Define viconCallback -----------------------%
function viconCallback_Quad(src, msg)
    global x_1 y_1 z_1 pitch_1 roll_1 yaw_1;
    x_1 = msg.Transform.Translation.X+2;
    y_1 = msg.Transform.Translation.Y+2;
    z_1 = msg.Transform.Translation.Z;
    
    qx = msg.Transform.Rotation.X;
    qy = msg.Transform.Rotation.Y;
    qz = msg.Transform.Rotation.Z;
    qw = msg.Transform.Rotation.W;
    ans = quat2eul([qw qx qy qz]);
    
    % default is ZYX -> YPR 
    yaw_1 = ans(1);
    pitch_1 = ans(2);
    roll_1 = ans(3);
end
%---------------------------- END ------------------------------------%