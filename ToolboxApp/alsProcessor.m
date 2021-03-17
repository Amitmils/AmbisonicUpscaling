classdef alsProcessor < BaseProcess
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = alsProcessor()
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.args('early reflections') = 20;
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R)
            [s,fs] = als(s,fs,roomDim,sourcePos,arrayPos,R,obj.args("early reflections"));
        end
    end
end

