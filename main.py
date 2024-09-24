from utils import *
import soundfile as sf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import time 
from optimizer import optimizer

if __name__ == '__main__':


    method = 'GD_lagrange_multi'#'GD_lagrange_multi' 'SQP'

    tau = 1024 #number of samples
    num_bins = 1
    SH_type = 'real'

    # Encode Signal
    sh_order_input = 1
    upscale_order = 3

    filename_list = [r"data/sound_files/female_speech.wav",r"data/sound_files/female_speech.wav"]
    th_list=[math.radians(45),math.radians(70)]
    ph_list = [math.radians(129),math.radians(220)]
    anm_t_list=list()
    start = 0
    end = 0
    for file,th,ph in zip(filename_list[start:end+1],th_list[start:end+1],ph_list[start:end+1]):
        s, fs = sf.read(file)
        s=s/np.sqrt(np.mean(s**2))
        anm_t_list.append(encode_signal(s,sh_order_input,ph,th,plot=False,type=SH_type))
    shortest_len = min([anm_t.shape[0] for anm_t in anm_t_list])
    anm_t = anm_t_list[0]
    for i in range(len(anm_t_list)):
        anm_t = anm_t + anm_t_list[i][:shortest_len,:]

    P = 162
    points= generate_sphere_points(P,plot=False)
    P_th = points[:,1]
    P_ph = points[:,2]
    Y_p = create_sh_matrix(sh_order_input,zen=P_th,azi=P_ph,type=SH_type)

    #project on 192 points
    projected_values = anm_t[0,:] @ np.conj(Y_p) / s[0]
    # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")

    # divide to subbands
    DS = 2
    if num_bins>1:
        anm_t_subbands = divide_anm_t_to_sub_bands(anm_t,fs,num_bins=num_bins-2,low_filter_center_freq=1,DS=DS)
    else:
        anm_t_subbands=np.expand_dims(anm_t,axis=0)
        anm_t_subbands=anm_t_subbands[:,::DS,:]

    # divide to time windows
    anm_t_subbands_windowed = divide_anm_t_to_time_windows(anm_t_subbands,window_length=tau) #[Window, band pass k ,time samples,SH coeff]
    num_windows = anm_t_subbands_windowed.shape[0]


    s_subbands = np.zeros((num_windows,num_bins,P,tau)) #num_windows, num_bins | P = num of SH coeff | tau = num of samples
    opt = optimizer(Y_p, alpha = 1,method=method)
    for band in range(num_bins):
        print(f"band {band}")
        for window in range(num_windows):
            start = time.time()
            print(f"window {window}")
            Bk = anm_t_subbands_windowed[window,band,:,:].T
            s_subbands[window,band,:,:],Dk = opt.optimize(Bk,D_prior = Dk if window > 0 else None)#imoptimize(Bk,Y_p,alpha = 1,D_prior = Dk if window > 0 else None )
            print(f"{time.time()-start} secs")


    s_windowed = np.sum(s_subbands, axis=1)
    s_dict = s_windowed.transpose(1, 0, 2).reshape(P, tau * num_windows)
    # plot_on_sphere([P_th,P_ph],s_dict[:,0]/s[0],title=f"Signal in Plane Wave Dictonary\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=s_dict[:,0]/s[0],title=f"Signal in Plane Wave Dictonary\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")


    Y_p_tag = create_sh_matrix(upscale_order,zen=P_th,azi=P_ph,type=SH_type)
    anm_upscaled = Y_p_tag @ s_dict[:,0] 
    s_upscaled = anm_upscaled @ np.conj(Y_p_tag)/s[0]
    # plot_on_sphere([P_th,P_ph],s_upscaled,title=f"Upscaled Signal N={upscale_order}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=s_upscaled,title=f"Upscaled Signal N={upscale_order}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")

    plt.show()





