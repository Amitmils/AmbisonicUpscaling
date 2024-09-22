import spaudiopy as spa
import numpy as np
from py_bank.filterbanks import EqualRectangularBandwidth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def cart2sph(cart_coords):
    x = cart_coords[:, 0]
    y = cart_coords[:, 1]
    z = cart_coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return np.column_stack((r, theta, phi))

def sph2cart(sph_coords):
    if not(isinstance(sph_coords, np.ndarray)):
        sph_coords = np.stack(sph_coords).T
    if sph_coords.shape[1] == 3:
        r = sph_coords[:, 0]
    else:
        r = np.ones(sph_coords.shape[0])
    th = sph_coords[:, -2]
    phi = sph_coords[:, -1]
    x = r * np.sin(th) * np.cos(phi)
    y = r * np.sin(th) * np.sin(phi)
    z = r * np.cos(th)
    return np.column_stack((x, y, z))

def create_sh_matrix(N,azi,zen,type='complex'):
    azi = azi.reshape(-1)
    zen = zen.reshape(-1)
    return spa.sph.sh_matrix(N_sph=N,azi=azi,zen=zen,sh_type=type).transpose()

def fft_anm_t(anm_t,fs):
    NFFT = 2 ** np.ceil(np.log2(anm_t.shape[0])).astype(int)  # Equivalent of nextpow2 in MATLAB
    anm_f = np.fft.fft(anm_t, NFFT, axis=0)  # Perform FFT along the rows (axis=0)

    # Remove negative frequencies
    anm_f = anm_f[:NFFT // 2 + 1, :]  # Keep only the positive frequencies

    # Vector of frequencies
    fVec = np.fft.fftfreq(NFFT, 1/fs)  # Create frequency vector
    fVec_pos = fVec[:NFFT // 2 + 1]  # Keep only positive frequencies
    return anm_f,fVec_pos

def divide_anm_t_to_sub_bands(anm_t,fs,num_bins,low_filter_center_freq,DS=2):
    #signal is size [num_samples,(ambi Order+1)^2]
    anm_t = anm_t[::DS]
    fs = fs / DS

    high_filter_center_freq = fs / 2  # centre freq. of highest filter
    num_samples,filter_length = anm_t.shape  # filter bank length
    erb_bank = EqualRectangularBandwidth(filter_length, fs, num_bins, low_filter_center_freq, high_filter_center_freq)
    anm_t_subbands = np.zeros((num_bins+2,num_samples,filter_length)) # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
    for time_sample in range(num_samples):
        erb_bank.generate_subbands(anm_t[time_sample])
        anm_t_subbands[:,time_sample,:] = erb_bank.subbands.T

    #[pass band k,t,SH_coeff]
    return anm_t_subbands

def divide_anm_t_to_time_windows(anm_t,window_length):
    #signal is size [band pass k ,time samples,(ambi Order+1)^2]
    num_samples = anm_t.shape[1]
    anm_t_padded = np.pad(anm_t, ((0, 0),(0, window_length - num_samples % window_length),(0, 0)), mode='constant', constant_values=0) 
    windowed_anm_t = np.array_split(anm_t_padded, anm_t_padded.shape[1] // window_length,axis=1)
    windowed_anm_t = np.stack(windowed_anm_t)
    return windowed_anm_t

def encode_signal(s,sh_order,ph,th,type = 'complex',plot=False):
    y = spa.sph.sh_matrix(N_sph=sh_order,azi=ph,zen=th,sh_type=type)
    if plot:
        debug = np.ones((1,16))
        debug = debug * (4 * np.pi) / (debug.shape[1] + 1)**2
        spa.plot.sh_coeffs(y,cbar=False) #Mirrored when use complex (Why?)
    encoded_signal = s.reshape(-1,1) @ y * (4 * np.pi) / (sh_order + 1)**2
    print("Energy of Encoded Signal:", np.mean(encoded_signal*np.conj(encoded_signal)))
    return encoded_signal


import numpy as np
from scipy.spatial import ConvexHull

# Define the 12 vertices of the icosahedron
def icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ])
    # Normalize vertices to lie on the sphere
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return vertices

# Subdivide triangle face into smaller triangles
def subdivide(vertices, faces, n):
    """
    Subdivides each triangle face of the icosahedron into smaller triangles.
    
    Parameters:
    vertices - Vertices of the current mesh
    faces - Triangle faces as indices of vertices
    n - Number of subdivisions
    
    Returns:
    new_vertices - The vertices after subdivision
    new_faces - The faces (triangles) after subdivision
    """
    def midpoint(v1, v2):
        return (v1 + v2) / 2

    new_vertices = list(vertices)
    midpoint_cache = {}
    
    def add_vertex(v):
        # Normalize the vertex to project onto the sphere
        v = v / np.linalg.norm(v)
        new_vertices.append(v)
        return len(new_vertices) - 1
    
    def get_midpoint_index(v1_idx, v2_idx):
        smaller_idx = min(v1_idx, v2_idx)
        larger_idx = max(v1_idx, v2_idx)
        key = (smaller_idx, larger_idx)
        
        if key not in midpoint_cache:
            mid = midpoint(vertices[v1_idx], vertices[v2_idx])
            midpoint_cache[key] = add_vertex(mid)
        
        return midpoint_cache[key]
    
    new_faces = []
    for tri in faces:
        v0, v1, v2 = tri
        
        a = get_midpoint_index(v0, v1)
        b = get_midpoint_index(v1, v2)
        c = get_midpoint_index(v2, v0)
        
        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])
    
    return np.array(new_vertices), np.array(new_faces)

# Generate P points on the sphere
def generate_sphere_points(P,plot):
    # Starting icosahedron
    vertices = icosahedron_vertices()
    faces = ConvexHull(vertices).simplices

    # Estimate how many subdivisions we need to get at least P points
    # Number of vertices after subdivision ~ (n_subdivisions^2 * initial_faces)
    n_faces = len(faces)
    n_subdivisions = round(np.sqrt(P/n_faces) + 0.5) #num points grows as ~(n_subdivisions ** 2) per face (Asymptotic growth)


    # Subdivide the icosahedron
    for i in range(n_subdivisions):
        vertices, faces = subdivide(vertices, faces, i)
        ## Plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=100, label='Vertices')
        # for simplex in faces:
        #     triangle = vertices[simplex]
        #     ax.add_collection3d(Poly3DCollection([triangle], color='cyan', edgecolor='k', linewidths=1, alpha=0.7))
        # plt.show()

    points = np.array(vertices[:P])
    if plot:
        spa.plot.hull(spa.decoder.get_hull(*points.T))
        plt.title(f"{len(points)} Points on Sphere")
    return cart2sph(points)

def plot_on_sphere(points,values,title=""):
    cart_points = sph2cart(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(cart_points[:,0], cart_points[:,1], cart_points[:,2], c=np.real(values), cmap='viridis', s=50)  # s is point size
    plt.colorbar(sc, label='Value')  # Add colorbar to show value scale
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

def plot_on_2D(azi,zen,values,title=""):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    azimuths_mollweide = azi
    zeniths_mollweide = np.pi / 2 - zen # Mollweide projection takes latitude, so shift by pi/2
    # Plot points with color representing the value at each point
    sc = ax.scatter(azimuths_mollweide, zeniths_mollweide, c=np.real(values), cmap='viridis', s=50)
    plt.title(title,pad=80)
    plt.colorbar(sc, label='Value')

    # Modify y-axis tick labels to range from 0 to 180 degrees, with steps of 15
    yticks_degrees = np.arange(0, 181, 15)  # From 0 to 180, with steps of 15 degrees
    yticks_radians = np.radians(90 - yticks_degrees)  # Convert to radians and shift by 90 to match Mollweide's latitude format

    # Apply ticks and labels
    ax.set_yticks(yticks_radians)
    ax.set_yticklabels(yticks_degrees)



