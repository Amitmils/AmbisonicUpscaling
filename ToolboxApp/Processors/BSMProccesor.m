classdef BSMProccesor < BaseProcess
        
    properties
        
    end
    
    methods
        function obj = BSMProccesor()
            %% define the specific arguments for the target function.
            % the signal, frequency, room dimensions, source position,mic
            % array position and the walls reflection defined globaly and
            % recieved in the process method.
            
            %each of this arguments need to be binded to a gui compnent in
            %Toolbox.mlapp:startupFcn(app) function see README.md or
            %startupFcn comment for furthur explanation.
            
            obj.args('arrayType') = 'Semi_Circular';              % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
            obj.args('rigidArray') = 'Rigid';                     % 0 - open array, 1 - rigid array     
            obj.args('r_array') = 0.1;              % array radius in meters
            obj.args('M') = 6;                      % number of mics of array
            obj.args('N_PW') = 15;                  % SH order of synthesized sound field
            obj.args('HRTFpath') = 'ToolboxApp/data/earoHRIR_KU100_Measured_2702Lebedev.mat';
            obj.args('headRotation') = false;
            obj.args('rot_idx') = 1;            
            
        end
        
        function [p_BSM_mag_t, fs]= process(obj, s, fs, roomDim, sourcePos, arrayPos, R)
            %% Process and return the signal.
            % use the global room parameters and the processor agruments to
            % process the signal. you can either use an external function
            % or do everything in place.
            
            switch obj.args('arrayType')
                case 'Spherical'
                    array_type_num = 0;
                case 'Semi_Circular'
                    array_type_num = 1;
                case 'Fully_Circular'                
                    array_type_num = 2;
            end
            switch obj.args('rigidArray')
                case 'Open'
                    rigidArray_num = 0;
                case 'Rigid'
                    rigidArray_num = 1;
            end
            
            [p_BSM_mag_t, fs] = BSM_script(s, fs, roomDim, sourcePos, arrayPos,...
                array_type_num, rigidArray_num, R, obj.args('r_array'),...
                obj.args('M'), obj.args('HRTFpath'), obj.args('N_PW'),...
                obj.args('headRotation'), obj.args('rot_idx'));
        end
    end
end

