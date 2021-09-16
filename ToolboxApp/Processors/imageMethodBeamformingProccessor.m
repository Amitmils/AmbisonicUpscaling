classdef imageMethodBeamformingProccessor < BaseProcess
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = imageMethodBeamformingProccessor()
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.args('N')= 4;
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R)
            [s,fs] = image_method_dpd_music_beamforming(s,fs,roomDim,sourcePos,arrayPos,R,obj.args("N"));
        end
    end
end

