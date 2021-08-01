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
            obj.args('rot_ang') =  30;
            obj.args('headRotation')=true;
            obj.args('LPF')=true;
            obj.args('Fc')=2;
            obj.args('Width')=2;
        end
        
        function [s_blt,fs_blt]= process(obj,s, fs,roomDim,sourcePos,head_center_Pos,R)
            % Process and return the signal.
            % use the global room parameters and the processor agruments to
            % process the signal. you can either use an external function
            % or do everything in place.
            
%             save_path = "/Users/orberebi/Desktop/exp_03/";
%             mkdir(save_path)
%             if obj.args('LPF')
%                 bilateral_name = save_path + "/LPF_N"+num2str(obj.args('N_bilat'))+"_"+num2str(obj.args('rot_ang'))+".wav";
%             else
%                 bilateral_name = save_path + "/blt_N"+num2str(obj.args('N_bilat'))+"_"+num2str(obj.args('rot_ang'))+".wav";
%             end

            
            
            %head size position and orientation
            %----------------------------------
            th_0_l = (pi/180)*(90);                             %left ear position
            ph_0_l = (pi/180)*(90);                             %left ear position
            th_0_r = (pi/180)*(90);                             %right ear position
            ph_0_r = (pi/180)*(270);                            %right ear position
            head_vec = [obj.args('head_radius'),th_0_l,ph_0_l,th_0_r,ph_0_r];

            [x0,y0,z0]=s2c(head_vec(2),head_vec(3),head_vec(1));
            recPosL = [x0,y0,z0] + head_center_Pos;
            [x0,y0,z0]=s2c(head_vec(4),head_vec(5),head_vec(1));
            recPosR = [x0,y0,z0] + head_center_Pos;

            
            disp('geting the left/right anmt...')
            tic
            [anm_t_L,~] = calc_room_anm_t(s, fs,roomDim,sourcePos,recPosL,...
                R,obj.args('N_bilat'));
            [anm_t_R,fs_blt] = calc_room_anm_t(s, fs,roomDim,sourcePos,recPosR,...
                R,obj.args('N_bilat'));
%             [anm_t_center,fs_blt] = calc_room_anm_t(s, fs,roomDim,sourcePos,head_center_Pos,...
%                 R,obj.args('N_bilat'));
            toc
            
            rot_ang_rad = (360 - obj.args('rot_ang'))*(pi/180);
            

            disp("anmt -> anmk")
            tic
            % Transform from time to frequency domain
            %make length of x even, and calculate frequency range
            nfft = max([size(anm_t_L,1) size(anm_t_R,1)]);
            nfft = 2^nextpow2(nfft);
            fftDim = 1;

            anmk_l = fft(anm_t_L,nfft,fftDim);
            anmk_l = anmk_l.';          %SH X freq
            anmk_l = anmk_l(:,1:end/2+1);

            anmk_r = fft(anm_t_R,nfft,fftDim);
            anmk_r = anmk_r.';          %SH X freq
            anmk_r = anmk_r(:,1:end/2+1);

            c = soundspeed();               % speed of sound [m/s]
            f=linspace(0,fs_blt/2,nfft/2+1);    % frequency range
            w=2*pi*f;                       % radial frequency
            k=w/c;                          %wave number
            kr=k*head_vec(1);               %k*head_radius
            toc

            if obj.args('headRotation')
                Fc = obj.args('Fc')*10^3;
                Width = obj.args('Width')*10^3;
                disp('Rotate anms...')
                tic
                [anm_l_k_A,anm_r_k_A] = rotate_anms(anmk_l,anmk_r,...
                    rot_ang_rad,obj.args('N_bilat'),head_vec,kr,...
                    obj.args('LPF'),Fc,Width,fs_blt);
                toc
            else
                anm_l_k_A = anmk_l;
                anm_r_k_A = anmk_r;
            end

            disp('Calc p(t) from HRTF and anm(k)...')
            tic
            [s_blt,fs_blt] = Binuaural_reproduction_bilateral_ambisonics...
                (anm_l_k_A,anm_r_k_A, fs_blt,obj.args('HRTFpath'),obj.args('N_bilat'));
            
            s_blt(size(anm_t_L, 1) + 1:end,:) = []; %cut the tail

            toc
            
%             audiowrite(bilateral_name,s_blt,fs_blt); %save signal to path

            disp("Done!")
        end
    end
end

