import numpy as np
import matplotlib.pyplot as plt


def plotting_style():
    """Plotting syle

    Define plotting style for the entire project

    """
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18

    plt.style.use('default')

    plt.rc('text', usetex=True)
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('font', serif='Times New Roman')  # fontsize of the figure title


def plot_reconstruction_2d(data, datarec, Fop, dx, f, ks, vel):
    D = Fop * data
    Drec = Fop * datarec
    nt = data.shape[1]

    fig, axs = plt.subplots(2, 3, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 0].imshow(data.T, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xlabel('Offset (m)')
    axs[0, 0].set_ylabel('TWT (s)')
    axs[0, 1].imshow(datarec.T, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    axs[0, 1].set_title('Reconstructed')
    axs[0, 1].set_xlabel('Offset (m)')
    axs[0, 2].imshow(data.T - datarec.T, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    axs[0, 2].set_title('Error')
    axs[0, 2].set_xlabel('Offset (m)')

    axs[1, 0].imshow(np.fft.fftshift(np.abs(D).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 0].plot(f / vel, f, 'w'), axs[1, 0].plot(f / vel, -f, 'w')
    axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 0].set_ylim(50, 0)
    axs[1, 0].set_xlabel('Wavenumber (1/m)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    axs[1, 1].imshow(np.fft.fftshift(np.abs(Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 1].plot(f / vel, f, 'w'), axs[1, 1].plot(f / vel, -f, 'w')
    axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 1].set_ylim(50, 0)
    axs[1, 1].set_xlabel('Wavenumber (1/m)')
    axs[1, 2].imshow(np.fft.fftshift(np.abs(D - Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 2].plot(f / vel, f, 'w'), axs[1, 2].plot(f / vel, -f, 'w')
    axs[1, 2].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 2].set_ylim(50, 0)
    axs[1, 2].set_xlabel('Wavenumber (1/m)')
    plt.tight_layout()
