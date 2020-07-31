mov = struct('cdata', [],...
                        'colormap', []);

%What is the Name of your AVI File?
Title='William_Levi_Peaks_View';                  
vidObj = VideoWriter(Title);
%What is the Frame Rate?
vidObj.FrameRate=14.1727;
open(vidObj)     

for j=1:count
%Plot, set the axis limits you want to maintain and then drawnow update.    
mesh(psi_H_save(:,:,j))
hold on
view([24,23])
xlim([0,42])
ylim([0,42])
hold off
    drawnow update
mov= getframe(gcf);
    
    writeVideo(vidObj,mov)
 
     clear mov
    mov = struct('cdata', [],...
                       'colormap', []);
end
%Close the file when you're done
close(vidObj)
