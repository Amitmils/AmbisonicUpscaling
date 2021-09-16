%% BSMgrid.m
% Returns the directions of assumed sources in BSM 
%% Inputs:
% source_distribution : (scalar) 0 - nearly uniform (t-design), 1 - spiral nearly uniform
% Q                   : (scalar) Assumed number of sources
%% Outputs:
% th_BSMgrid_vec      :  (Q x 1) elevation angles 
% ph_BSMgrid_vec      :  (Q x 1) azimuth angles 

function [th_BSMgrid_vec, ph_BSMgrid_vec] = BSMgrid(source_distribution, Q)

    switch source_distribution
        case 0
            % calculate steering vectors with nearly uniform sampling
            N_NU_sources_direction = min( [floor(sqrt(Q)) - 1, 10] );
            [~, th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.uniform_sampling_extended(N_NU_sources_direction);
            th_BSMgrid_vec = th_BSMgrid_vec.';
            ph_BSMgrid_vec = ph_BSMgrid_vec.';
            ph_BSMgrid_vec = ph_BSMgrid_vec + pi;
            %plot_sampling(th_grid_vec, ph_grid_vec);
            %plot_sampling_hammer(th_grid_vec, ph_grid_vec);
        case 1
            % calculate steering vectors with spiral nearly uniform sampling
            [~, th_BSMgrid_vec, ph_BSMgrid_vec] = SpiralSampleSphere_w_weights(Q);
            %plot_sampling(th_grid_vec, ph_grid_vec);
            %plot_sampling_hammer(th_grid_vec, ph_grid_vec);

    end
end





