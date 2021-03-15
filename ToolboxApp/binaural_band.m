arrayPos = [7 5 1.7];
Room = [15.5 9.8 7.5];
N_array= 4;
r_array= 0.042;
path = 'data\';
files = ["bass.wav" "accompaniment.wav" "drums.wav" "other.wav" "vocals.wav"];
sourcePos=[[7,3,1.7],[7,3,1.7],[7,5,1.7],[7,3,1.7],[7 5 1.7]];
res =0;
for k=1:length(files)
    [x,fs] = audioread(strcat(path,files(k)));
    [x,fs] = pwd_binaural_reproduction(sum(x,2),fs,Room,sourcePos(k),arrayPos,0.9,N_array,r_array);
    res=res+x;
end
soundsc(res, fs);
