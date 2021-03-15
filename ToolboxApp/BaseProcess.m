classdef BaseProcess
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        args = containers.Map;
    end
    methods(Abstract)
        [s,fs]= process(obj);
    end
    methods
        
        
        function bind(obj,component,argName)
     
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            component.Value = obj.args(argName);
            function callback(~,src,~)
                obj.args(argName) = src.Value;
            end
            component.ValueChangedFcn ={@callback};
        end
    end
end

