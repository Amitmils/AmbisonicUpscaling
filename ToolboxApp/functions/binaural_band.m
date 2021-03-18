arrayPos = [7 5 1.7];
Room = [15.5 9.8 7.5];
N_array= 4;
r_array= 0.042;
path = 'ToolboxApp\data\sounds\';
files = ["bass.wav" "accompaniment.wav" "drums.wav" "other.wav" "vocals.wav"];
sourcePos=[[7,3,1.7];[7,3,1.7];[9,5,1.7];[7,7,1.7];[7 5 1.7]];
res =0;
for k=1:length(files)
    [sound,fs] = audioread(strcat(path,files(k)));
    sound = sum(sound,2);
    x =[];
%     for i=1:44100:length(sound)
%         disp(sourcePos(1+(k-1)*3:k*3))
%         [tmp,fs] = pwd_binaural_reproduction(sound(i:min(i+44100,length(sound))),fs,Room,sourcePos(1+(k-1)*3:k*3),arrayPos,0.9,N_array,r_array,'ToolboxApp/data/earoHRIR_KU100_Measured_2702Lebedev.mat');
%         x = cat(1,x,tmp);
%     end
    [x,fs] = pwd_binaural_reproduction(sound,fs,Room,sourcePos(k,:),arrayPos,0.9,N_array,r_array,'ToolboxApp/data/earoHRIR_KU100_Measured_2702Lebedev.mat');
    res=res+x;
end
% fs = 44100;
soundsc(res, fs);
