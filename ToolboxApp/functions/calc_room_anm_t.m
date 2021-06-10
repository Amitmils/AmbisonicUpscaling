function anm_t = calc_room_anm_t(s, fs,roomDim,sourcePos,arrayPos,R,N_PW)
%CALC_ROOM_ANM_T Summary of this function goes here
%   Detailed explanation goes here
max_length =0
for k=1:length(s)
    [~,colnum]=size(s{k});
    if colnum>1
        s{k}=sum(s{k},2);
    end
    if length(s{k})>max_length
        max_length = length(s{k});
    end
end
anm_t = zeros(max_length,256);
for k=1:length(s)

[hnm, ~] = image_method.calc_rir(fs, roomDim, sourcePos{k}, arrayPos, R, {}, {"array_type", "anm", "N", N_PW});
% T60 = RoomParams.T60(hnm(:,1), fs); 
% CriticalDist = RoomParams.critical_distance_diffuse(roomDim, R);

% figure; plot((0:size(hnm,1)-1)/fs, real(hnm(:,1))); % plot the RIR of a00
% xlabel('Time [sec]');
anm_t(1:length(s{k}),:) = anm_t(1:length(s{k}),:) +  fftfilt(hnm, s{k});

% anm_t = hnm; 
clear hnm
end
end

