import math
import torchaudio

from typing import Callable, Optional
from warnings import warn
from scipy import signal
from scipy.ndimage import sobel


import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.compliance import kaldi

import numpy as np
from scipy.fftpack import ifft



class LPC(torch.nn.Module):

    __constants__ = ['sample_rate', 'num_ceps', 'pre_emph',
                     'pre_emph_coeff', 'win_type', 'win_len', 'win_hop', 'do_rasta', 'dither']

    def __init__(self,
                 num_ceps: int = 13,
                 sample_rate: int = 16000,
                 pre_emph: int = 0,
                 pre_emph_coeff: float = 0.97,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 #win_type: str = 'hann',
                 win_len: float = 0.025,
                 win_hop: float = 0.01,
                 do_rasta: bool = True,
                 dither: int = 1,
                 melkwargs: Optional[dict] = None) -> None:
        super(LPC, self).__init__()
        self.num_ceps = num_ceps
        self.sample_rate = sample_rate
        self.pre_emph = pre_emph
        self.pre_emph_coeff = pre_emph_coeff
        #self.win_type = win_type
        self.win_len = win_len
        self.win_hop = win_hop
        self.do_rasta = do_rasta
        self.dither = dither
        self.window = window_fn(int(self.win_len*self.sample_rate))


    def forward(self, waveform: Tensor) -> Tensor:

        if self.pre_emph:
            waveform = self.pre_emphasis(waveform, self.pre_emph_coeff)

        power_spectrum = self.powspec(waveform, sample_rate=self.sample_rate,
                                    win_type=self.window, win_len=self.win_len, win_hop=self.win_hop, dither=self.dither)

        auditory_spectrum = self.audspec(power_spectrum, self.sample_rate)
        nbands = auditory_spectrum.shape[1]

        if self.do_rasta:
            # put in log domain
            log_auditory_spectrum = torch.log(auditory_spectrum)
            # next do rasta filtering
            # (21,435)
            rasta_filtered_log_auditory_spectrum = self.rasta_filter(
                log_auditory_spectrum)
            # do inverse log
            # (21,435)
            auditory_spectrum = torch.exp(rasta_filtered_log_auditory_spectrum)

            post_processing_spectrum, _ = self.postaud(auditory_spectrum, self.sample_rate / 2)

            lpcs = self.do_lpc(x=post_processing_spectrum, model_order=self.num_ceps)

            #lpcs = lpcs.T
            batch, first, second = lpcs.shape
            #lpcs = lpcs.view(batch, second, first)
            return lpcs[:, :self.num_ceps , :]



    def do_lpc(self, x, model_order=8):
        batch, nbands, nframes = x.shape
        ncorr = 2 * (nbands - 1)
        R = torch.zeros((batch ,ncorr, nframes))

        R[: ,0:nbands, :] = x
        for i in range(nbands - 1):
            R[:, i + nbands - 1, :] = x[:, nbands - (i + 1), :]

        R = R.detach().cpu().numpy()
        for i in range(batch):
            R[i,:,:] = ifft(R[i,:,:].T).real.T
        r = R[:,0:nbands, :]

        y = np.ones((batch, nframes, model_order + 1))
        e = np.zeros((nframes, 1))

        finalR = np.ones((batch, model_order+1, nframes))
        if model_order == 0:
            for i in range(nframes):
                _, e_tmp, _ = self.LEVINSON(r[:, i],
                                                  model_order,
                                                  allow_singularity=True)
                e[i, 0] = e_tmp
        else:
            for b in range(batch):
                for i in range(nframes):
                    y_tmp, e_tmp, _ = self.LEVINSON(r[b,:, i],
                                                          model_order,
                                                          allow_singularity=True)
                    y[b, i, 1:model_order + 1] = y_tmp
                    e[i, 0] = e_tmp
                #tm1 = y[b,:,:].T
                #tm2 = (np.tile(e.T, (model_order + 1, 1)) + 1e-8)
                finalR[b,:,:] = y[b,:,:].T / (np.tile(e.T, (model_order + 1, 1)) + 1e-8)

        return torch.as_tensor(finalR,dtype=torch.float32).cuda()





    def pre_emphasis(self, waveform, pre_emph_coeff):
        return torch.cat((waveform[:,0].unsqueeze(1),waveform[:,1:]-pre_emph_coeff*waveform[:,:-1]),1)






    def powspec(self, waveform, sample_rate=16000, nfft=512, win_type='hann', win_len=0.025, win_hop=0.01, dither=1):
        # convert win_len and win_hop from seconds to samples
        win_length = int(win_len * sample_rate)
        hop_length = int(win_hop * sample_rate)
        fft_length = torch.pow(2, torch.ceil(torch.log2(torch.tensor(win_len * sample_rate))))

        #[5,257,202]
        waveform = torch.stft(input=waveform, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=win_type.cuda(),
                              center=True, pad_mode="reflect", onesided=True, return_complex=True)


        #测试实现是否正确，和torchaudio保持一致
        #[5,201,162]
        # waveform = torch.stft(input=waveform, n_fft=400, hop_length=200, win_length=win_length, window=win_type.cuda(),
        #                       center=True, pad_mode="reflect", onesided=True, return_complex=True)

        pow_waveform = torch.abs(waveform)**2
        if dither:
            pow_waveform = pow_waveform + win_length
        return pow_waveform





    def audspec(self,
                p_spectrum,
                fs=16000,
                nfilts=0,
                low_freq=0,
                high_freq=0,
                sumpower=1,
                bwidth=1):

        # if nfilts == 0:
        #     np.ceil(hz2bark(fs / 2)) + 1
        if high_freq == 0:
            high_freq = fs / 2
        nfreqs = p_spectrum.shape[1]
        nfft = (int(nfreqs) - 1) * 2

        #(21,400)
        wts = self.fft2barkmx(nfft, fs, nfilts, bwidth, low_freq, high_freq)

        wts = wts[:, 0:nfreqs]

        wts = torch.as_tensor(wts, dtype=torch.float32).cuda()

        if sumpower:
            aspectrum = torch.matmul(wts, p_spectrum)
        else:
            aspectrum = torch.matmul(wts, torch.sqrt(p_spectrum))**2
        return aspectrum





    def hz2bark(self, f):
        """
        Convert Hz frequencies to Bark acoording to Wang, Sekey & Gersho, 1992.

        Args:
            f (np.array) : input frequencies [Hz].

        Returns:
            (np.array): frequencies in Bark [Bark].
        """

        return 6. * np.arcsinh(f / 600.)





    def fft2barkmx(self, nfft, fs, nfilts=0, bwidth=1, low_freq=0, high_freq=0):

        if high_freq == 0:
            high_freq = fs / 2
        min_bark = self.hz2bark(low_freq)
        nyqbark = self.hz2bark(high_freq) - min_bark

        if nfilts == 0:
            nfilts = int(np.add(np.ceil(nyqbark), 1))

        if not isinstance(nfilts, int):
            print("nfilts Error")
            return

        if not isinstance(nfft, int):
            print("nfft Error")
            return
        #[21,400]
        wts = np.zeros((nfilts, nfft))
        step_barks = nyqbark / (nfilts - 1)
        binbarks = self.hz2bark((fs / nfft) * np.arange(0, nfft / 2 + 1))

        for i in range(nfilts):
            f_bark_mid = min_bark + i * step_barks
            lof = binbarks - f_bark_mid - 0.5
            hif = binbarks - f_bark_mid + 0.5
            wts[i, 0:nfft // 2 + 1] = 10**np.minimum(
                0,
                np.minimum(hif, -2.5 * lof) / bwidth)
        return wts




    def rasta_filter(self, x):
        """
        % y = rastafilt(x)
        %
        % rows of x = critical bands, cols of x = frame
        % same for y but after filtering
        %
        % default filter is single pole at 0.94
        """
        numer = np.arange(-2, 3)
        numer = (-1 * numer) / np.sum(numer * numer)
        denom = np.array([1, -0.94])

        zi = signal.lfilter_zi(numer, 1)

        x = x.detach().cpu().numpy()
        y = np.zeros((x.shape))
        count = 0
        for xitem in x:
            for i in range(xitem.shape[0]):
                y1, zi = signal.lfilter(numer, 1, xitem[i, 0:4], axis=0, zi=zi * xitem[i, 0])
                y1 = y1 * 0
                y2, _ = signal.lfilter(numer, denom, xitem[i, 4:xitem.shape[1]], axis=0, zi=zi)
                y[count, i, :] = np.append(y1, y2)
            count += 1
        return torch.tensor(y, dtype=torch.float32).cuda()





    def bark2hz(self, fb):
        """
        Convert Bark frequencies to Hz.

        Args:
            fb (np.array) : input frequencies [Bark].

        Returns:
            (np.array)  : frequencies in Hz [Hz].
        """
        return 600. * np.sinh(fb / 6.)






    def postaud(self, x, fmax, fb_type='bark', broaden=0):
        """
        do loudness equalization and cube root compression
            - x = critical band filters
            - rows = critical bands
            - cols = frames
        """
        _, nbands, nframes = x.shape
        nfpts = int(nbands + 2 * broaden)

        bandcfhz = self.bark2hz(np.linspace(0, self.hz2bark(fmax), nfpts))


        bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

        fsq = np.power(bandcfhz, 2)
        ftmp = np.add(fsq, 1.6e5)
        eql = np.multiply((fsq / ftmp) ** 2, (fsq + 1.44e6) / (fsq + 9.61e6))


        # z = np.multiply(np.tile(eql, (nframes, 1)).T, x)
        # z = np.power(z, 0.33)
        tmp = torch.as_tensor(np.tile(eql,(nframes,1)).T).cuda()
        #x = x.cuda()
        z = torch.mul(x, tmp)
        z = torch.pow(z, 0.33)

        if broaden:
            y = np.zeros((z.shape[0] + 2, z.shape[1]))
            y[0, :] = z[0, :]
            y[1:nbands + 1, :] = z
            y[nbands + 1, :] = z[z.shape[0] - 1, :]
        else:
            # y = np.zeros((z.shape[0], z.shape[1]))
            # y[0, :] = z[1, :]
            # y[1:nbands - 1, :] = z[1:z.shape[0] - 1, :]
            # y[nbands - 1, :] = z[z.shape[0] - 2, :]
            y = torch.zeros((z.shape[0] ,z.shape[1], z.shape[2]))
            y[:,0,:] = z[:,1,:]
            y[:,1:nbands-1,:] = z[:,1:z.shape[1]-1:]
            y[:, nbands-1,:]= z[:,z.shape[1]-2,:]
        #y=[5,23,162]
        return y, eql

    def LEVINSON(slef, r, order=None, allow_singularity=False):
        """
        Levinson-Durbin recursion.
        Find the coefficients of a length(r)-1 order autoregressive linear process

        Args:
            r                (array) : autocorrelation sequence of length N + 1
                                       (first element being the zero-lag autocorrelation)
            order              (int) : requested order of the autoregressive coefficients.
                                       Default is N.
            allow_singularity (bool) : Other implementations may be True (e.g., octave)
                                       Default is False.
        Returns:
            * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
            * the prediction errors
            * the `N` reflections coefficients values

        Note:

            This algorithm solves the set of complex linear simultaneous equations
            using Levinson algorithm.

        .. math::
            \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
            \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)
        where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
        :math:`T_0, T_1, \dots ,T_M`.

        Note:
            Solving this equations by Gaussian elimination would
            require :math:`M^3` operations whereas the levinson algorithm
            requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.
        This is equivalent to solve the following symmetric Toeplitz system of
        linear equations

        .. math::
            \left( \begin{array}{cccc}
            r_1 & r_2^* & \dots & r_{n}^*\\
            r_2 & r_1^* & \dots & r_{n-1}^*\\
            \dots & \dots & \dots & \dots\\
            r_n & \dots & r_2 & r_1 \end{array} \right)
            \left( \begin{array}{cccc}
            a_2\\
            a_3 \\
            \dots \\
            a_{N+1}  \end{array} \right)
            =
            \left( \begin{array}{cccc}
            -r_2\\
            -r_3 \\
            \dots \\
            -r_{N+1}  \end{array} \right)
        where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
        :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
        a vector of autocorrelation coefficients where lag 0 is the first
        element :math:`r_1`.
        .. raw::python
            import numpy; from spectrum import LEVINSON
            T = numpy.array([3., -2+0.5j, .7-1j])
            a, e, k = LEVINSON(T)
        """
        # from numpy import isrealobj
        T0 = np.real(r[0])
        T = r[1:]
        M = len(T)

        if order is None:
            M = len(T)
        else:
            assert order <= M, 'order must be less than size of the input data'
            M = order

        realdata = np.isrealobj(r)
        if realdata is True:
            A = np.zeros(M, dtype=float)
            ref = np.zeros(M, dtype=float)
        else:
            A = np.zeros(M, dtype=complex)
            ref = np.zeros(M, dtype=complex)

        P = T0

        for k in range(0, M):
            save = T[k]
            if k == 0:
                temp = -save / P
            else:
                # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
                for j in range(0, k):
                    save = save + A[j] * T[k - j - 1]
                temp = -save / P
            if realdata:
                P = P * (1. - temp ** 2.)
            else:
                P = P * (1. - (temp.real ** 2 + temp.imag ** 2))
            if P <= 0 and allow_singularity == False:
                raise ValueError("singular matrix")
            A[k] = temp
            ref[k] = temp  # save reflection coeff at each step
            if k == 0:
                continue

            khalf = (k + 1) // 2
            if realdata is True:
                for j in range(0, khalf):
                    kj = k - j - 1
                    save = A[j]
                    A[j] = save + temp * A[kj]
                    if j != kj:
                        A[kj] += temp * save
            else:
                for j in range(0, khalf):
                    kj = k - j - 1
                    save = A[j]
                    A[j] = save + temp * A[kj].conjugate()
                    if j != kj:
                        A[kj] = A[kj] + temp * save.conjugate()

        return A, P, ref