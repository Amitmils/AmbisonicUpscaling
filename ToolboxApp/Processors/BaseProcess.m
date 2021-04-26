classdef BaseProcess
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        args = containers.Map;
    end
    methods(Abstract)
        [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R);
    end
    methods
        
        
        function res=bind(obj,component,argnm,r,callb)
            
            function callback(src,e,r)
                if r
                    src.Value = round(src.Value);
                end
                obj.args(argnm) = src.Value;
                 if exist('callb','var')
                    callb(src.Value)
                end
            end
            if exist('r','var')
                component.ValueChangedFcn ={@callback,true};
            else
                component.ValueChangedFcn ={@callback,false};
            end
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            component.Value = obj.args(argnm);
            res=true;
        end
        
    end
end


