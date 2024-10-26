from utils import *
import soundfile as sf
import matplotlib.pyplot as plt
import math
import time 
from optimizer import optimizer
import scipy.io


if __name__ == '__main__':


    method = 'GD_lagrange_multi'# 'GD_lagrange_multi' 'SQP' 'SLS'
    constraint_tol = 0 #only for GD_lagrange_multi
    
    grid_type = 'lebedev' #lebedev or manual

    tau = 1024 #number of samples
    num_bins = 45
    SH_type = 'real'

    suffix = 'LALALA'

    # Encode Signal
    sh_order_input = 1
    upscale_order = 3


    max_num_windows = 20

    filename_list = [r"data/sound_files/male_speech.wav",r"data/sound_files/female_speech.wav"]
    th_list=[90,90] #degrees
    ph_list = [45,0] #degrees
    anm_t_list=list()
    s_list=list()
    fs_list=list()
    y_list=list()
    start_idx = 0
    end_idx = 1

    #encode signals
    for file,th,ph in zip(filename_list[start_idx:end_idx+1],th_list[start_idx:end_idx+1],ph_list[start_idx:end_idx+1]):
        anm_t,s,fs,y = encode_signal(file,sh_order_input,th=math.radians(th),ph=math.radians(ph),plot=False,type=SH_type)
        anm_t_list.append(anm_t)
        s_list.append(s)
        fs_list.append(fs)
        y_list.append(y)
    smallest_fs = min(fs_list)

    fs_list = np.array(fs_list)
    #resample such that all have the same sampling rate
    for i in (fs_list!=smallest_fs).nonzero()[0]:
        assert fs_list[i] % smallest_fs == 0,f"Error in fs for file {filename_list[i]} (min is {smallest_fs} and it has {fs_list[i]})"
        ds_rate = int(fs_list[i] / smallest_fs)
        anm_t_list[i] = anm_t_list[i][::ds_rate]
    fs = smallest_fs

    shortest_len = min([anm_t.shape[0] for anm_t in anm_t_list])

    anm_t = 0
    for i in range(len(anm_t_list)):
        anm_t = anm_t + anm_t_list[i][:shortest_len,:]

    if grid_type == 'lebedev':
        P = 2702
        lebedev = scipy.io.loadmat('Lebvedev2702.mat')
        P_th = lebedev['th'].reshape(-1) #rad
        P_ph = lebedev['ph'].reshape(-1) #rad
        P_ph = (P_ph + np.pi) % (2 * np.pi) - np.pi #wrap angles to [-pi,pi]
    else:
        P = 162
        points= generate_sphere_points(P,plot=False)
        P_th = points[:,1]
        P_ph = points[:,2]

    Y_p = create_sh_matrix(sh_order_input,zen=P_th,azi=P_ph,type=SH_type)

    #project on 192 points
    projected_values = anm_t[0,:] @ np.conj(Y_p)
    # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {th_list[start_idx:end_idx+1]} \n$\\phi$ = {ph_list[start_idx:end_idx+1]}")

    # divide to subbands
    DS = 2
    if num_bins>1:
        anm_t_subbands = divide_anm_t_to_sub_bands(anm_t,fs,num_bins=num_bins-2,low_filter_center_freq=1,DS=DS)
    else:
        anm_t_subbands=np.expand_dims(anm_t,axis=0)[:,::DS,:]

    # divide to time windows
    anm_t_subbands_windowed = divide_anm_t_to_time_windows(anm_t_subbands,window_length=tau) #[Window, band pass k ,time samples,SH coeff]
    num_windows = min(anm_t_subbands_windowed.shape[0],max_num_windows)


    Y_p_tag = create_sh_matrix(upscale_order,zen=P_th,azi=P_ph,type=SH_type) #upscaling matrix
    target_theta,target_phi =np.array([math.radians(90),math.radians(90)]) , np.array([math.radians(45),math.radians(0)])
    mask =  np.logical_or.reduce(np.sqrt((P_th[:,None] - target_theta)**2 + (P_ph[:,None] - target_phi)**2)< math.radians(5), axis=1)
    if mask is not None:
        dummy = np.zeros_like(mask)
        dummy[mask] = 1
        plot_on_2D(azi=P_ph,zen=P_th,values=dummy,title=f"Spatial Mask {i//tau}\n$\\theta$ = {th_list[start_idx:end_idx+1]} \n$\\phi$ = {ph_list[start_idx:end_idx+1]}")
        # plt.show()

    s_subbands = np.zeros((num_windows,num_bins,P,tau)) #num_windows, num_bins | P = num of SH coeff | tau = num of samples
    opt = optimizer(Y_p, alpha = 1,constraint_tol=constraint_tol,method=method)
    for band in range(num_bins):
        print(f"band {band}")
        for window in range(num_windows):
            start = time.time()
            print(f"window {window}")
            Bk = anm_t_subbands_windowed[window,band,:,:].T
            s_subbands[window,band,:,:],Dk = opt.optimize(Bk,mask,D_prior=None)#imoptimize(Bk,Y_p,alpha = 1,D_prior = Dk if window > 0 else None )
            # tmp = s_subbands[window,band,:,:]
            # plot_on_2D(azi=P_ph,zen=P_th,values=tmp[:,0],title=f"Band {band} in Plane Wave Dictonary\n$\\theta$ = {th_list[start_idx:end_idx+1]} \n$\\phi$ = {ph_list[start_idx:end_idx+1]}")
            # plt.show()
            print(f"{time.time()-start} secs")
    file = os.path.join('data/output',f"{'_'.join([os.path.basename(file_name).split('.wav')[0] for file_name in filename_list[start_idx:end_idx+1]])}_FBbins_{num_bins}_grid_{grid_type}_th_{'_'.join([str(th) for th in th_list[start_idx:end_idx+1]])}_ph_{'_'.join([str(ph) for ph in ph_list[start_idx:end_idx+1]])}_N_input_{sh_order_input}_{suffix}")
    save_sparse_matrix(filename = file,matrix=s_subbands)
    print(f"Saved to {file}")


    if False:
        #Show results
        s_windowed = np.sum(s_subbands, axis=1)
        ideal_constraint_loss = list() #when we have 1 activated source
        algo_constraint_loss = list()
        algo_constraint_loss_v2 = list()

        test = list()
        for i in range(min(0,1)*tau,min(num_windows,1)* tau,tau):
            s_dict = s_windowed.transpose(1, 0, 2).reshape(P, tau * num_windows)
            plot_on_2D(azi=P_ph,zen=P_th,values=s_dict[:,i]/np.sign(s[i]),title=f"Signal in Plane Wave Dictonary Window {i//tau}\n$\\theta$ = {th_list[start_idx:end_idx+1]} \n$\\phi$ = {ph_list[start_idx:end_idx+1]}")
            P_index = np.argmin((P_ph - math.radians(ph_list[0]))**2 + (P_th - math.radians(th_list[0]))**2)
            ideal_constraint_loss.append(np.sum((Y_p[:,P_index] * s[::DS][i].reshape(-1,1)- anm_t[::DS][i])**2))
            algo_constraint_loss.append(np.sum((Y_p@ s_dict[:,i].reshape(-1,1) - anm_t[::DS][i].reshape(-1,1))**2))
            # print(f"Algo Constraint {(Y_p@ s_dict[:,i].reshape(-1,1) - anm_t[i,:].reshape(-1,1))}")
            # print(f"Ideal Constraint {Y_p[:,P_index] * s[i].reshape(-1,1)- anm_t[i,:]}")
            anm_upscaled = Y_p_tag @ s_dict[:,i]
            s_upscaled = anm_upscaled @ np.conj(Y_p_tag)/np.sign(s[i])
            # plot_on_2D(azi=P_ph,zen=P_th,values=s_upscaled,title=f"Upscaled Signal N={upscale_order} Window Window {i//tau}\n$\\theta$ = {th_list[start_idx:end_idx+1]} \n$\\phi$ = {ph_list[start_idx:end_idx+1]}")
        # plt.figure()
        # plt.plot(ideal_constraint_loss,label='Ideal',marker='o')
        # plt.plot(algo_constraint_loss,label='Algorithm',marker='o')
        # plt.legend()
        # plt.title("Constraint Loss")
        # plt.xlabel("Window")
        # plt.ylabel("Loss")
        plt.show()





