classdef binauralProccesor < BaseProcess
        
    properties
        
    end
    
    methods
        function obj = binauralProccesor()
            %% define the specific arguments for the target function.
            % the signal, frequency, room dimensions, source position,mic
            % array position and the walls reflection defined globaly and
            % recieved in the process method.
            
            %each of this arguments need to be binded to a gui compnent in
            %Toolbox.mlapp:startupFcn(app) function see README.md or
            %startupFcn comment for furthur explanation.
                
            obj.args('N_array')= 4;
            obj.args('r_array')= 0.042;
            obj.args('ShOrder')=15;
            obj.args('HRTFpath') = 'ToolboxApp/data/earoHRIR_KU100_Measured_2702Lebedev.mat';
            obj.args('rot_idx') = 1;
            obj.args('headRotation')=false;
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,arrayPos,R)
            %% Process and return the signal.
            % use the global room parameters and the processor agruments to
            % process the signal. you can either use an external function
            % or do everything in place.
            
            anm_t = calc_room_anm_t(s, fs,roomDim,sourcePos,arrayPos,R,obj.args('ShOrder'));
            [s,fs] = pwd_binaural_reproduction(anm_t,fs,obj.args("N_array"),obj.args("r_array"),obj.args('HRTFpath'),obj.args('ShOrder'),obj.args('headRotation'),360-obj.args('rot_idx'));
        end
    end
end

