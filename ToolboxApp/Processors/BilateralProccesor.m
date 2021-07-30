classdef BilateralProccesor < BaseProcess
        
    properties
        
    end
    
    methods
        function obj = BilateralProccesor()
            % define the specific arguments for the target function.
            % the signal, frequency, room dimensions, source position,mic
            % array position and the walls reflection defined globaly and
            % recieved in the process method.
            
            %each of this arguments need to be binded to a gui compnent in
            %Toolbox.mlapp:startupFcn(app) function see README.md or
            %startupFcn comment for furthur explanation.
                
            obj.args('N_bilat')= 4;
            obj.args('head_radius')= 0.0875;
            obj.args('HRTFpath') = 'ToolboxApp/data/earoHRIR_KU100_Measured_2702Lebedev.mat';
            obj.args('rot_ang') = 30;
            obj.args('headRotation')=true;
        end
        
        function [s,fs]= process(obj,s, fs,roomDim,sourcePos,head_center_Pos,head_radius,refCoef)
            % Process and return the signal.
            % use the global room parameters and the processor agruments to
            % process the signal. you can either use an external function
            % or do everything in place.
            
            %head size position and orientation
            %----------------------------------
            th_0_l = (pi/180)*(90);                             %left ear position
            ph_0_l = (pi/180)*(90);                             %left ear position
            th_0_r = (pi/180)*(90);                             %right ear position
            ph_0_r = (pi/180)*(270);                            %right ear position
            head_vec = [head_radius,th_0_l,ph_0_l,th_0_r,ph_0_r];

            [x0,y0,z0]=s2c(head_vec(2),head_vec(3),head_vec(1));
            recPosL = [x0,y0,z0] + head_center_Pos;
            [x0,y0,z0]=s2c(head_vec(4),head_vec(5),head_vec(1));
            recPosR = [x0,y0,z0] + head_center_Pos;

            
            [anm_t_L,fs] = calc_room_anm_t(s, fs,roomDim,sourcePos,recPosL,...
                refCoef,obj.args('N_bilat'));
            [anm_t_R,fs] = calc_room_anm_t(s, fs,roomDim,sourcePos,recPosR,...
                refCoef,obj.args('N_bilat'));
            
            %[s,fs] = pwd_binaural_reproduction(anm_t,fs,obj.args("N_array"),...
            %    obj.args("r_array"),obj.args('HRTFpath'),obj.args('ShOrder'),...
            %    obj.args('headRotation'),360-obj.args('rot_idx'));
            
            [s,fs] = Bilateral_Ambisonics_binaural_reproduction(anm_t_L, anm_t_R, fs, obj.args('N_bilat'), head_vec, HRTFpath,rot_ang);
        end
    end
end

