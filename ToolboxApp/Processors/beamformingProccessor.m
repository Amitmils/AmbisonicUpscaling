classdef beamformingProccessor < BaseProcess
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = beamformingProccessor()
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.args('N_array')= 4;
            obj.args('r_array')= 0.042;
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R)
            [s,fs] = pwd_dpd_music_beamforming(s,fs,roomDim,sourcePos,arrayPos,R,obj.args("N_array"),obj.args("r_array"));
        end
    end
end

