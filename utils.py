import spaudiopy as spa
import soundfile as sf
import os
import numpy as np
from py_bank.filterbanks import EqualRectangularBandwidth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import torchaudio
import torch
from signal_info import signal_info


def plot_loss_tensor(loss_tensor):
    loss_tensor = loss_tensor.cpu().detach()
    num_iter,num_windows,num_bins = loss_tensor.shape
    avg_loss_per_freq = torch.mean(loss_tensor,dim=1)
    for i in range(num_bins):
        plt.plot(avg_loss_per_freq[:,i],label=f"Freq {i}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Per Frequency Bin")

def save_sparse_matrix(matrix, filename):
    """
    Save a sparse matrix to a file.

    Parameters:
    matrix : Sparse matrix to save.
    filename : Name of the file to save the matrix.

    Returns:
    None
    """

    # Get the indices and values of nonzero elements
    indices = np.array(np.nonzero(matrix)).T
    values = matrix[tuple(indices.T)]

    # Save indices and values to a file in .npz format
    return np.savez_compressed(
        filename, indices=indices, values=values, shape=matrix.shape
    )


def load_sparse_matrix(filename):
    # Load the data from the .npz file
    data = np.load(filename + ".npz")
    indices = torch.tensor(data["indices"])
    values = torch.tensor(data["values"])
    shape = tuple(torch.tenosr(data["shape"]))

    # Reconstruct the sparse array
    sparse_array = torch.zeros(shape, dtype=values.dtype)
    sparse_array[tuple(indices.T)] = values

    return sparse_array


def cart2sph(cart_coords):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    x (numpy.ndarray): Array of x coordinates.
    y (numpy.ndarray): Array of y coordinates.
    z (numpy.ndarray): Array of z coordinates.

    Returns:
    numpy.ndarray: Array of spherical coordinates (r, theta, phi).
    """
    x = cart_coords[:, 0]
    y = cart_coords[:, 1]
    z = cart_coords[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z / r)
    phi = torch.arctan2(y, x)
    return torch.stack((r, theta, phi),dim=1)


def sph2cart(sph_coords):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    [r,th,phi]
    r (numpy.ndarray): Array of radial distances.
    th (numpy.ndarray): Array of polar angles in radians.
    phi (numpy.ndarray): Array of azimuthal angles in radians.

    Returns:
    numpy.ndarray: Array of Cartesian coordinates (x, y, z).
    """
    if not (isinstance(sph_coords, torch.tensor)):
        sph_coords = torch.stack(sph_coords).T
    if sph_coords.shape[1] == 3:
        r = sph_coords[:, 0]
    else:
        r = torch.ones(sph_coords.shape[0])
    th = sph_coords[:, -2]
    phi = sph_coords[:, -1]
    x = r * torch.sin(th) * torch.cos(phi)
    y = r * torch.sin(th) * torch.sin(phi)
    z = r * torch.cos(th)
    return torch.stack((x, y, z),dim=1)


def create_sh_matrix(N, azi, zen, type="real"):
    """
    Create a spherical harmonics matrix.

    Parameters:
    N (int): The order of the spherical harmonics.
    azi (numpy.ndarray): Array of azimuthal angles in radians.
    zen (numpy.ndarray): Array of zenith angles in radians.
    type (str, optional): Type of spherical harmonics ('complex' or 'real'). Default is 'complex'.

    Returns:
    numpy.ndarray: The spherical harmonics matrix.
    """
    azi = azi.reshape(-1)
    zen = zen.reshape(-1)
    return torch.tensor(spa.sph.sh_matrix(N_sph=N, azi=azi.cpu().numpy(), zen=zen.cpu().numpy(), sh_type=type).transpose()).to(torch.float32)


def fft_anm_t(anm_t, fs):
    """
    Perform FFT on time-domain spherical harmonic coefficients.

    Parameters:
    anm_t (numpy.ndarray): Time-domain spherical harmonic coefficients.
    fs (float): Sampling frequency.

    Returns:
    numpy.ndarray: Frequency-domain spherical harmonic coefficients.
    """
    NFFT = 2 ** torch.ceil(torch.log2(anm_t.shape[0])).astype(
        int
    )  # Equivalent of nextpow2 in MATLAB
    anm_f = torch.fft.fft(anm_t, n=NFFT, dim=0)  # Perform FFT along the rows (axis=0)

    # Remove negative frequencies
    anm_f = anm_f[: NFFT // 2 + 1, :]  # Keep only the positive frequencies

    # Vector of frequencies
    fVec = torch.fft.fftfreq(NFFT, 1 / fs)  # Create frequency vector
    fVec_pos = fVec[: NFFT // 2 + 1]  # Keep only positive frequencies
    return anm_f, fVec_pos


def _resample(signal, org_sr, new_sr):
    """
    Resample a signal to a new sampling rate.

    Parameters:
    signal (numpy.ndarray): The input signal.
    org_sr (int): Original sampling rate of the signal.
    new_sr (int): New sampling rate.

    Returns:
    numpy.ndarray: The resampled signal.
    """
    if org_sr == new_sr:
        return signal
    else:
        resampler = torchaudio.transforms.Resample(orig_freq=org_sr, new_freq=new_sr).to(signal.device)
        resampled_signal = resampler(signal.unsqueeze(0)).squeeze()
        if isinstance(signal, np.ndarray):
            return resampled_signal.numpy().squeeze()
        elif isinstance(signal, torch.Tensor):
            return resampled_signal

        else:
            raise ValueError("Input signal must be a numpy array or a PyTorch tensor.")


def divide_anm_t_to_sub_bands(anm_t, fs, num_bins, low_filter_center_freq, DS=2):
    # signal is size [num_samples,(ambi Order+1)^2]
    anm_t = anm_t[::DS]
    fs = fs / DS

    high_filter_center_freq = fs / 2  # centre freq. of highest filter
    num_samples, num_coeff = anm_t.shape  # filter bank length
    erb_bank = EqualRectangularBandwidth(
        num_samples, fs, num_bins, low_filter_center_freq, high_filter_center_freq
    )
    anm_t_subbands = torch.zeros(
        (num_bins + 2, num_samples, num_coeff)
    )  # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
    for coeff in range(num_coeff):
        erb_bank.generate_subbands(anm_t[:, coeff].numpy())
        anm_t_subbands[:, :, coeff] = torch.tensor(erb_bank.subbands.T)

    # [pass band k,t,SH_coeff]
    return anm_t_subbands

def encode_signal(
    signal : signal_info,
    sh_order,
    ph=None,
    th=None,
    type="complex",
    plot=False,
    normalize_signal=True,
):

    try:
        s = signal.signal
        fs = signal.sr
        th = signal.th
        ph = signal.ph
    except:
        raise f"signal must be of type signal_info"
    if normalize_signal:
        s = s / torch.sqrt(torch.mean(s**2))
    y = torch.tensor(spa.sph.sh_matrix(N_sph=sh_order, azi=ph, zen=th, sh_type=type)).to(torch.float32)

    if plot:
        debug = torch.ones((1, 16))
        debug = debug * (4 * torch.pi) / (debug.shape[1] + 1) ** 2
        spa.plot.sh_coeffs(y, cbar=False)  # Mirrored when use complex (Why?)
    encoded_signal = torch.matmul(s.reshape(-1, 1), y.reshape(1, -1))
    return encoded_signal, s, fs, y


# Define the 12 vertices of the icosahedron
def icosahedron_vertices():
    """
    Generate the vertices of an icosahedron.

    Returns:
    numpy.ndarray: Array of shape (12, 3) containing the Cartesian coordinates of the vertices.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]
    )
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

    return np.stack(new_vertices), np.stack(new_faces)


# Generate P points on the sphere
def generate_sphere_points(P, plot):
    """
    Generate points almost uniformly distributed on a sphere.

    Parameters:
    P (int): Number of points to generate.
    plot: flag to plot the points

    Returns:
    numpy.ndarray: Array of shape (num_points, 3) containing the Spherical coordinates of the points.
    """
    # Starting icosahedron
    vertices = icosahedron_vertices()
    faces = ConvexHull(vertices).simplices

    # Estimate how many subdivisions we need to get at least P points
    # Number of vertices after subdivision ~ (n_subdivisions^2 * initial_faces)
    n_faces = len(faces)
    n_subdivisions = round(
        np.sqrt(P / n_faces) + 0.5
    )  # num points grows as ~(n_subdivisions ** 2) per face (Asymptotic growth)

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

    points = torch.tensor(vertices[:P])
    if plot:
        spa.plot.hull(spa.decoder.get_hull(*points.T))
        plt.title(f"{len(points)} Points on Sphere")
    return cart2sph(points)


def plot_on_sphere(points, values, title=""):
    """
    Plot values on a 3D sphere.

    Parameters:
    points ([azi,zen]): Spherical coordinates of the points
    values (numpy.ndarray): Array of values to plot.
    title (str, optional): Title of the plot. Default is an empty string.

    Returns:
    None
    """
    cart_points = sph2cart(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        cart_points[:, 0],
        cart_points[:, 1],
        cart_points[:, 2],
        c=torch.real(values),
        cmap="viridis",
        s=50,
    )  # s is point size
    plt.colorbar(sc, label="Value")  # Add colorbar to show value scale
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)


def create_sin_wave(freq, duration=10.0, fs=48000, output_dir="data/sound_files"):
    """
    Generate a sine wave and save it as a WAV file.

    Parameters:
    freq (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the sine wave in seconds.
    fs (int): Sampling frequency in Hz.
    output_dir (str): Directory to save the generated WAV file.

    Returns:
    None
    """
    # Time array
    t = torch.linspace(0, duration, steps=int(fs * duration), endpoint=False)

    # Generate the sine wave
    sine_wave = 1 + 0.5 * torch.sin(
        2 * torch.pi * freq * t
    )  # Amplitude of 0.5 to avoid clipping
    # Save as a WAV file
    sf.write(os.path.join(output_dir, f"{freq}Hz_sine_wave.wav"), sine_wave, fs)


def plot_on_2D(azi, zen, values, title="", normalize=True):
    """
    Plot values on a 2D Mollweide projection.

    Parameters:
    azi (numpy.ndarray): Array of azimuthal angles in radians.
    zen (numpy.ndarray): Array of zenith angles in radians.
    values (numpy.ndarray): Array of values to plot.
    title (str, optional): Title of the plot. Default is an empty string.
    normalize (bool, optional): Whether to normalize the values. Default is True.

    Returns:
    None
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    azimuths_mollweide = azi
    zeniths_mollweide = (
        torch.pi / 2 - zen
    )  # Mollweide projection takes latitude, so shift by pi/2
    # Plot points with color representing the value at each point

    sc = ax.scatter(
        azimuths_mollweide.cpu(), zeniths_mollweide.cpu(), c=torch.real(values).cpu(), cmap="viridis", s=25
    )
    plt.title(title, pad=80)
    plt.colorbar(sc, label="Value")

    # Modify y-axis tick labels to range from 0 to 180 degrees, with steps of 15
    yticks_degrees = torch.arange(0, 181, 15)  # From 0 to 180, with steps of 15 degrees
    yticks_radians = torch.deg2rad(
        90 - yticks_degrees
    )  # Convert to radians and shift by 90 to match Mollweide's latitude format

    # Apply ticks and labels
    ax.set_yticks(yticks_radians.cpu().numpy())
    ax.set_yticklabels(yticks_degrees.cpu().numpy())
