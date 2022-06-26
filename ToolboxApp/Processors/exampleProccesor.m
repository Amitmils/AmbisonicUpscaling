classdef exampleProccesor < BaseProcess
        
    properties
        
    end
    
    methods
        function obj = exampleProccesor()
            %% define the specific arguments for the target function.
            % the signal, frequency, room dimensions, source position,mic
            % array position and the walls reflection defined globaly and
            % recieved in the process method.
            
            %each of this arguments need to be binded to a gui compnent in
            %Toolbox.mlapp:startupFcn(app) function see README.md or
            %startupFcn comment for furthur explanation.
                
            obj.args('example1') = 20;
            obj.args('example2') = 0.42;
            
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R)
            %% Process and return the signal.
            % use the global room parameters and the processor agruments to
            % process the signal. you can either use an external function
            % or do everything in place.
            [s,fs] = example_function(s,fs,roomDim,sourcePos,arrayPos,R,obj.args("example1"),obj.args("example1"));
        end
    end
end

