from utils import *
import soundfile as sf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math


def optimize(Bk,Y_p,sh_order,P,alpha,D_prior=None,dim_reduction=False):
    def objective(Omega_k):
        Omega_k = Omega_k.reshape(Omega_k0.shape)
        res = np.sum(np.sqrt(np.sum(Omega_k**2,axis=1))) #TODO verify this is the correct objective function
        return res

    def constraint(Omega_k):
        Omega_k = Omega_k.reshape(Omega_k0.shape)
        return (Y_p@Omega_k - Uk@Lambda_k).flatten()

    # The constraint dictionary
    con = {'type': 'eq', 'fun': constraint}

    if dim_reduction:
        Uk, Lk, Vkt = np.linalg.svd(Bk)
        Lambda_k = np.diag(Lk)# diagonal matrix of singular values
        Omega_k0 = np.zeros((P,Lambda_k.shape[1])) # init Omega_k

        res = minimize(objective, Omega_k0.flatten(), constraints=con, method='SLSQP')
        Omega_k_opt = res.x.reshape(Omega_k0.shape)
        Dk = Omega_k_opt @ np.linalg.pinv(Uk @ Lambda_k)
        if D_prior is not None:
            Dk = alpha * Dk + (1 - alpha) * D_prior
        Sk = Dk @ Bk
    else:
        Omega_k0 = np.zeros((P,Bk.shape[1]))
        Uk = Bk
        Lambda_k = np.eye(Bk.shape[1])
        res = minimize(objective, Omega_k0.flatten(), constraints=con, method='SLSQP')
        Sk = res.x.reshape(Omega_k0.shape)
        Dk = None

    return Sk,Dk


if __name__ == '__main__':

    tau = 1024 #number of samples
    num_bins = 1
    SH_type = 'real'

    # Encode Signal
    sh_order_input = 1
    filename = r'data/sound_files/female_speech.wav'
    #TODO Problem with singular matrices when coefficients are 0 due to perfect simitry of the signal
    th = math.radians(45)
    ph = math.radians(45)
    s, fs = sf.read(filename)
    s=s/np.sqrt(np.mean(s**2))
    anm_t = encode_signal(s,sh_order_input,ph,th,plot=False,type=SH_type)

    P = 162
    points= generate_sphere_points(P,plot=False)
    P_th = points[:,1]
    P_ph = points[:,2]
    Y_p = create_sh_matrix(sh_order_input,zen=P_th,azi=P_ph,type=SH_type)

    #project on 192 points
    projected_values = anm_t[0,:] @ np.conj(Y_p) / s[0]
    plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")

    # divide to subbands
    DS = 2
    if num_bins>1:
        anm_t_subbands = divide_anm_t_to_sub_bands(anm_t,fs,num_bins=num_bins,low_filter_center_freq=1,DS=DS)
        num_bins+=2 #Include low and high pass bands for perfect reconstruction
    else:
        anm_t_subbands=np.expand_dims(anm_t,axis=0)
        anm_t_subbands=anm_t_subbands[:,::DS,:]

    # divide to time windows
    anm_t_subbands_windowed = divide_anm_t_to_time_windows(anm_t_subbands,window_length=tau) #[Window, band pass k ,time samples,SH coeff]
    num_windows = anm_t_subbands_windowed.shape[0]



    s_subbands = np.zeros((num_windows,num_bins,P,tau)) #num_windows, num_bins | P = num of SH coeff | tau = num of samples
    for band in range(num_bins):
        print(f"band {band}")
        for window in range(num_windows):
            print(f"window {window}")
            Bk = anm_t_subbands_windowed[window,band,:,:].T
            s_subbands[window,band,:,:],Dk = optimize(Bk, Y_p, sh_order_input, P, alpha = 1, D_prior = Dk if window > 0 else None, dim_reduction=True)
            break
    
    s_windowed = np.sum(s_subbands, axis=1)
    s_dict = s_windowed.transpose(1, 0, 2).reshape(P, tau * num_windows)
    max_point = np.argmax((s_dict[:,0]))
    plot_on_sphere([P_th,P_ph],s_dict[:,0]/s[0],title=f"Signal in Plane Wave Dictonary\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=s_dict[:,0]/s[0],title=f"Signal in Plane Wave Dictonary\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")


    upscale_order = 3
    Y_p_tag = create_sh_matrix(upscale_order,zen=P_th,azi=P_ph,type=SH_type)
    anm_upscaled = Y_p_tag @ s_dict[:,0] /s[0]
    s_upscaled = anm_upscaled @ np.conj(Y_p_tag)
    plot_on_sphere([P_th,P_ph],s_upscaled,title=f"Upscaled Signal N={upscale_order}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
    plot_on_2D(azi=P_ph,zen=P_th,values=s_upscaled,title=f"Upscaled Signal N={upscale_order}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")

    plt.show()





