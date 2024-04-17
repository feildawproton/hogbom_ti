# -- GLOBAL IMPORTS -- 
from astropy.io import fits
import numpy as np
import copy
import math
import taichi as ti
#import scikits.cuda.fft as fft
#import skcuda.fft as fft

# -- PLEASE TREAT AS CONST --
ARC_TO_RAD = math.pi / 180. / 3600.
HWIDTH     = 3
NCGF       = 12
GRID_KERN_WIDTH = 6
GRID_KERN_SMPLS = 24

def calc_spheroidal_wave(eta, width, alpha):
    # https://www.researchgate.net/publication/234365312_Optimal_Gridding_of_Visibility_Data_in_Radio_Interferometry
    # should I get this? https://onlinelibrary.wiley.com/doi/book/10.1002/0471221570
    
    #twoalp = 2*alpha
    if np.abs(eta) > 1:
        print('bad eta value!')
    if (2*alpha < 1 or 2*alpha > 4):
        print('bad alpha value!')
    if (width < 4 or width > 8):
        print('bad width value!')

    etalim = np.array([1., 1., 0.75, 0.775, 0.775], dtype=np.float32)
    nnum   = np.array([5, 7, 5, 5, 6], np.int8)
    ndenom = np.array([3, 2, 3, 3, 3], np.int8) 
    p      = np.array([ 
                       [ 
                        [5.613913E-2,-3.019847E-1, 6.256387E-1,-6.324887E-1, 3.303194E-1, 0.0, 0.0],
                        [6.843713E-2,-3.342119E-1, 6.302307E-1,-5.829747E-1, 2.765700E-1, 0.0, 0.0],
                        [8.203343E-2,-3.644705E-1, 6.278660E-1,-5.335581E-1, 2.312756E-1, 0.0, 0.0],
                        [9.675562E-2,-3.922489E-1, 6.197133E-1,-4.857470E-1, 1.934013E-1, 0.0, 0.0],
                        [1.124069E-1,-4.172349E-1, 6.069622E-1,-4.405326E-1, 1.618978E-1, 0.0, 0.0]
                       ],
                       [ 
                        [8.531865E-4,-1.616105E-2, 6.888533E-2,-1.109391E-1, 7.747182E-2, 0.0, 0.0],
                        [2.060760E-3,-2.558954E-2, 8.595213E-2,-1.170228E-1, 7.094106E-2, 0.0, 0.0],
                        [4.028559E-3,-3.697768E-2, 1.021332E-1,-1.201436E-1, 6.412774E-2, 0.0, 0.0],
                        [6.887946E-3,-4.994202E-2, 1.168451E-1,-1.207733E-1, 5.744210E-2, 0.0, 0.0],
                        [1.071895E-2,-6.404749E-2, 1.297386E-1,-1.194208E-1, 5.112822E-2, 0.0, 0.0]
                       ] 
                      ], dtype=np.float32)
    q = np.array([  
                  [ 
                   [1., 9.077644E-1, 2.535284E-1],
                   [1., 8.626056E-1, 2.291400E-1],
                   [1., 8.212018E-1, 2.078043E-1],
                   [1., 7.831755E-1, 1.890848E-1],
                   [1., 7.481828E-1, 1.726085E-1]
                  ],
                  [ 
                   [1., 1.101270   , 3.858544E-1],
                   [1., 1.025431   , 3.337648E-1],
                   [1., 9.599102E-1, 2.918724E-1],
                   [1., 9.025276E-1, 2.575337E-1],
                   [1., 8.517470E-1, 2.289667E-1]
                  ]
                 ], dtype=np.float32)

    i = int(width - 4)
    if(np.abs(eta) > etalim[i]):
        ip = 1
        x = eta*eta - 1
    else:
        ip = 0
        x = eta*eta - etalim[i]*etalim[i] 
    # numerator via Horner's rule 
    mnp  = nnum[i]-1
    num = p[ip,int(2*alpha),mnp]
    for j in np.arange(mnp):
        num = num*x + p[ip,int(2*alpha),mnp-1-j]
    # denominator via Horner's rule
    nq = ndenom[i]-1
    denom = q[ip,int(2*alpha),nq]
    for j in np.arange(nq):
        denom = denom*x + q[ip,int(2*alpha),nq-1-j]

    return  num/denom


def make_gridding_kernel(nsamp=2047, width=6, alpha=1.0):
    # https://lweb.cfa.harvard.edu/~pwilliam/miriad-python/docs/data-util.html
    # https://ui.adsabs.harvard.edu/abs/1999ASPC..180..127B/abstract
    # https://github.com/pkgw/miriad-python/blob/master/mirtask/util.py#L1072
    
    
    phi = np.zeros(shape=nsamp, dtype=np.float32)
    for ndx in range(nsamp):
        #x = (2*ndx-(nsamp-1))/(nsamp-1)
        eta = (2*ndx)/(nsamp-1) - 1
        phi[ndx] = (math.sqrt(1-eta*eta)**2*alpha)*calc_spheroidal_wave(eta=eta, width=width, alpha=alpha)
    return phi

def make_correction_kernel(nsamp=2047, width=6, alpha=1):
    dx = 2./nsamp
    i0 = nsamp/2+1
    phi = np.zeros(shape=nsamp, dtype=np.float32)
    for ndx in range(nsamp):
        eta = (ndx-i0+1)*dx
        phi[ndx] = calc_spheroidal_wave(eta=eta, width=width, alpha=alpha)
    return phi
    

@ti.kernel
def grid_BM_Vis_mirrored(du: float, u: ti.template(), v: ti.template(), real: ti.template(), imag: ti.template(), cgf: ti.template(),
                         bm: ti.template(), grid: ti.template(), cnt: ti.template()):
    nu = bm.shape[0]
    u0 = int(nu * 0.5)
    for x_ndx, y_ndx in bm:
        u_ndx = y_ndx
        v_ndx = x_ndx
        hfac  = 1

        if u_ndx < u0:
            u_ndx = nu - y_ndx
            v_ndx = nu - x_ndx
            hfac  = -1

        for vis_ndx in range(u.shape[0]):
            mu = u[vis_ndx]
            mv = v[vis_ndx]
            
            hflag = 1
            if mu < 0:
                hflag = -1
                mu = -1*mu
                mv = -1*mv
            
            uu  = mu/du+u0
            vv  = mv/du+u0
            cnu = ti.abs(u_ndx-uu)
            cnv = ti.abs(v_ndx-vv)


            if cnu < HWIDTH and cnv < HWIDTH:
                cnu_ndx    = int(ti.round(4.6*cnu+NCGF-0.5))
                cnv_ndx    = int(ti.round(4.6*cnv+NCGF-0.5))

                kernelval_at_ndx = cgf[cnu_ndx]*cgf[cnv_ndx]
                
                bm[x_ndx, y_ndx][0]   +=            kernelval_at_ndx
                grid[x_ndx, y_ndx][0] +=            kernelval_at_ndx*real[vis_ndx]
                grid[x_ndx, y_ndx][1] += hfac*hflag*kernelval_at_ndx*imag[vis_ndx]
                cnt[x_ndx, y_ndx]     += 1
                
            if (u_ndx-u0) < HWIDTH and (mu/du) < HWIDTH:
                mu  = -1*mu
                mv  = -1*mv
                uu  = mu/du+u0
                vv  = mv/du+u0
                cnu = ti.abs(u_ndx-uu)
                cnv = ti.abs(v_ndx-vv)
                if cnu < HWIDTH and cnv < HWIDTH:
                    cnu_ndx    = int(ti.round(4.6*cnu+NCGF-0.5))
                    cnv_ndx    = int(ti.round(4.6*cnv+NCGF-0.5))
                    
                    kernelval_at_ndx = cgf[cnu_ndx]*cgf[cnv_ndx]
                    
                    bm[x_ndx, y_ndx][0]   +=               kernelval_at_ndx
                    grid[x_ndx, y_ndx][0] +=               kernelval_at_ndx*real[vis_ndx]
                    grid[x_ndx, y_ndx][1] += hfac*-1*hflag*kernelval_at_ndx*imag[vis_ndx]

#ti.template() passes by reference
@ti.kernel
def apply_weights(count: ti.template(), briggs_wght: ti.f32, grid: ti.template()):
    for u_ndx, v_ndx in grid:
        cnt = count[u_ndx, v_ndx]
        if cnt != 0.0:
            weight = 1./ti.sqrt(1. + cnt*cnt/(briggs_wght*briggs_wght))
            grid[u_ndx, v_ndx][0] = grid[u_ndx, v_ndx][0]*weight
            grid[u_ndx, v_ndx][1] = grid[u_ndx, v_ndx][1]*weight

@ti.kernel
def shift(grid: ti.template(), out_grid: ti.template()):
    u_half = int(0.5*grid.shape[0])
    for u_ndx, v_ndx in grid:
        u_ndx_o = u_ndx-u_half
        v_ndx_o = v_ndx-u_half
        
        if u_ndx < u_half:
            u_ndx_o = u_half+u_ndx
        if v_ndx < u_half:
            v_ndx_o = u_half+v_ndx
        out_grid[u_ndx_o, v_ndx_o] = grid[u_ndx, v_ndx]

@ti.kernel
def correct_grid(grid: ti.template(), corr_kernel: ti.template()):
    nx = corr_kernel.shape[0]
    for x_ndx, y_ndx in grid:
        correction = corr_kernel[int(nx/2)]*corr_kernel[int(nx/2)]/(corr_kernel[x_ndx]*corr_kernel[x_ndx])
        grid[x_ndx,y_ndx][0] = grid[x_ndx,y_ndx][0]*correction
        grid[x_ndx,y_ndx][1] = grid[x_ndx,y_ndx][1]*correction
        
@ti.kernel
def trim(full_image: ti.template(), sub_image: ti.template()):
    diff_x = full_image.shape[0] - sub_image.shape[1]
    diff_y = full_image.shape[0] - sub_image.shape[1]
    x_offset = int(diff_x/2)
    y_offset = int(diff_y/2)
    for x_ndx, y_ndx in sub_image:
        sub_image[x_ndx,y_ndx][0] = full_image[x_ndx+x_offset, y_ndx+y_offset][0]
        sub_image[x_ndx,y_ndx][1] = full_image[x_ndx+x_offset, y_ndx+y_offset][1]
        
@ti.kernel
def normalize(field: ti.template()):
    field = ti.math.normalize(field)
    
        
# putting in an object so that we have a destructor, which is needed
@ti.data_oriented
class FitsFile(object):
    def __init__(self, path: str, use_ti: bool):
        print("attempting to open:", path)
        self.hdu_list = fits.open(path)
        print(self.hdu_list.info())  
        if type(self.hdu_list[1].data) == np.ndarray:
            print("this is an image")
            self.is_image = True
        else:
            print("this is not and image")
            self.is_image = False  
        self.use_ti = use_ti
                
    def __del__(self):
        #self.hdu_list.close()                        
        pass
    def convert_to_grid(self, grid_dim_len: int, pix_arcsecs: float, briggs_weight: float) -> np.ndarray:
        # -- 0.) IF ALREADY AN IMAGE, JUST RETURN IT --
        if self.is_image == True:
            return copy.deepcopy(self.hdu_list["SCI"].data)
        else:
            print("making grid")
            '''
              References:
                - gICLEAN by Katherine Rosenfeld and Nathan Sanders: http://nesanders.github.io/gICLEAN/ 
                - Chapter 10 in "Interferometry and Synthesis in Radio Astronomy" by Thompson, Moran, & Swenson
                - Daniel Brigg's PhD Thesis: http://www.aoc.nrao.edu/dissertations/dbriggs/ 
            '''
            ti.init()
            

            nx          = 2*grid_dim_len
            du          = 1. / (ARC_TO_RAD * pix_arcsecs * nx)          

            data     = self.hdu_list[0].data
            good_ndc = np.where(data.data[:,0,0,0,0,0,0] != 0)[0]
            
            
            
            good_u   = data.par("uu")[good_ndc]
            good_v   = data.par("vv")[good_ndc]       
            
            real = np.add(data.data[good_ndc,0,0,0,0,0,0], data.data[good_ndc,0,0,0,0,1,0])
            real = np.multiply(0.5, real)
            imag = np.add(data.data[good_ndc,0,0,0,0,0,1], data.data[good_ndc,0,0,0,0,1,1])
            imag = np.multiply(0.5, imag)
            
            freq = self.hdu_list[0].header['CRVAL4']
            u    = np.multiply(freq, good_u)
            v    = np.multiply(freq, good_v)
            
            u_start = int(0.5 * nx)
            
            u_max_abs  = np.max(np.fabs(u))
            v_max_abs  = np.max(np.fabs(v))
            u_max      = int(np.ceil(u_max_abs/du))
            v_max      = int(np.ceil(v_max_abs/du))
            u_stop     = u_start + u_max
            u_stop     = min(u_stop, nx)
            v_stop     = u_start + v_max
            v_stop     = min(v_stop, nx)
            
            print("u_max", u_max, "v_max", v_max)
            if u_max < nx or v_max < nx:
                print("WARNING! SPATIAL RESOLUTION TOO HIGH")
            print("need to handle nx larger than num samples. nx", nx, "u_stop", u_stop, "v stop", v_stop)

            
            cgf_np = make_gridding_kernel(nsamp=GRID_KERN_SMPLS, width=GRID_KERN_WIDTH, alpha=1.0)

            real_ti = ti.field(dtype=ti.f32, shape=real.shape)
            real_ti.from_numpy(real)
            imag_ti = ti.field(dtype=ti.f32, shape=imag.shape)
            imag_ti.from_numpy(imag)
            
            u_ti = ti.field(dtype=ti.f32, shape=u.shape)
            u_ti.from_numpy(u)
            v_ti = ti.field(dtype=ti.f32, shape=v.shape)
            v_ti.from_numpy(v)
            cgf_ti = ti.field(dtype=ti.f32, shape=cgf_np.shape)
            cgf_ti.from_numpy(cgf_np)
            
            grid_ti   = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
            cnts_ti   = ti.field(shape=(u_stop, v_stop), dtype=ti.int32)
            bm_ti     = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
            shifted_grid_ti = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
            shifted_bm_ti   = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
            
            # 1.) GRID
            # 1.1 and 1.3) bm and 1.6) grids
            grid_BM_Vis_mirrored(du=du, u=u_ti, v=v_ti, real=real_ti, imag=imag_ti, cgf=cgf_ti, 
                                    bm=bm_ti, grid=grid_ti, cnt=cnts_ti)
            
            # 1.2) apply weights to bm
            apply_weights(count=cnts_ti, briggs_wght=briggs_weight, grid=bm_ti)
            
            # 1.4) shift bm
            shift(grid=bm_ti, out_grid=shifted_bm_ti)
            
            # 1.5) weight vis grid
            apply_weights(count=cnts_ti, briggs_wght=briggs_weight, grid=grid_ti)
            
            # 1.7 ) shift grid
            shift(grid_ti, shifted_grid_ti)
            
            # replacing
            # plan = fft.Plan((self.nx,self.nx),np.complex64,np.complex64) 
            # fft.fft(shifted_bm, bm, plan)
            # i think the args are (in, out, plan)
            # https://cseweb.ucsd.edu/classes/wi15/cse262-a/static/cuda-5.5-doc/html/cufft/index.html which follows
            # http://www.fftw.org/fftw3.pdf
            # https://medium.com/codex/pycuda-the-fft-and-the-gerchberg-saxton-algorithm-35fb7bceb62f
            
            # 2.1)
            shifted_bm_np  = shifted_bm_ti.to_numpy()
            shifted_bm_np  = np.add(shifted_bm_np[:,:,0], np.multiply(shifted_bm_np[:,:,1], 1j)) 
            fft_shifted_bm = np.fft.fft2(shifted_bm_np, s=shifted_bm_np.shape)
    
            print(fft_shifted_bm.shape, type(fft_shifted_bm), fft_shifted_bm.dtype)
            
            # 2.2)
            fft_shifted_bm_uncmplx = np.zeros(shape=(fft_shifted_bm.shape[0], fft_shifted_bm.shape[1], 2), dtype=np.float32)
            fft_shifted_bm_uncmplx[:,:,0] = fft_shifted_bm.real
            fft_shifted_bm_uncmplx[:,:,1] = fft_shifted_bm.imag
            
            fft_shifted_bm_ti = ti.Vector.field(n=2, shape=fft_shifted_bm.shape, dtype=ti.float32)
            fft_shifted_bm_ti.from_numpy(fft_shifted_bm_uncmplx)
            fft_bm_ti = ti.Vector.field(n=2, shape=fft_shifted_bm.shape, dtype=ti.float32)
            
            shift(grid=fft_shifted_bm_ti, out_grid=fft_bm_ti)
            
            # 2.3)
            corr_kernel_np = make_correction_kernel(nsamp=nx, width=GRID_KERN_WIDTH)
            corr_kernel_ti = ti.field(shape=(nx), dtype=ti.float32)
            corr_kernel_ti.from_numpy(corr_kernel_np)
            
            correct_grid(grid=fft_bm_ti, corr_kernel=corr_kernel_ti)

            # 2.4)
            trimmed_beam_ti = ti.Vector.field(n=2, shape=(grid_dim_len, grid_dim_len), dtype=ti.float32)
            trim(fft_bm_ti, trimmed_beam_ti)
            
            trimmed_beam = trimmed_beam_ti.to_numpy()
            
            # 2.5)
            max_of_bm = np.max(trimmed_beam, axis=(0,1))
            normed_beam = np.multiply(trimmed_beam, np.divide(1., max_of_bm))
            
            # 2.6)
            shifted_grid_np  = shifted_grid_ti.to_numpy()
            shifted_grid_np  = np.add(shifted_grid_np[:,:,0], np.multiply(shifted_grid_np[:,:,1], 1j)) 
            fft_shifted_grid = np.fft.fft2(shifted_grid_np)
            
            #2.7)
            fft_shifted_grid_unc = np.zeros(shape=(fft_shifted_grid.shape[0], fft_shifted_grid.shape[1], 2), dtype=np.float32)
            fft_shifted_grid_unc[:,:,0] = fft_shifted_grid.real
            fft_shifted_grid_unc[:,:,1] = fft_shifted_grid.imag
            
            fft_shifted_grid_ti = ti.Vector.field(n=2, shape=fft_shifted_grid.shape, dtype=ti.float32)
            fft_shifted_grid_ti.from_numpy(fft_shifted_grid_unc)
            fft_grid_ti = ti.Vector.field(n=2, shape=fft_shifted_bm.shape, dtype=ti.float32)
            
            shift(grid=fft_shifted_grid_ti, out_grid=fft_grid_ti)
            
            # 2.8)                
            correct_grid(grid=fft_grid_ti, corr_kernel=corr_kernel_ti)
            
            # 2.9)
            trimmed_grid_ti = ti.Vector.field(n=2, shape=(grid_dim_len, grid_dim_len), dtype=ti.float32)
            trim(fft_grid_ti, trimmed_grid_ti)
            
            trimmed_grid = trimmed_grid_ti.to_numpy()
            
            # 2.10)
            normed_image = np.multiply(trimmed_grid, np.divide(1., max_of_bm))
            
            
            dirty_psf = normed_beam
            dirty_img = normed_image
            
            return dirty_psf, dirty_img
            
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    fits_file = FitsFile(path="demo_data/sim1.mickey.alma.out20.ms.fits", use_ti=True)
    grid_dim_len = 64 # characteristic size of the image, size will be grid_dim_len*grid_dim_len
    pix_arcsecs  = 5.12 / grid_dim_len
    briggs_weight = 1e3
    dirty_psf, dirty_img = fits_file.convert_to_grid(grid_dim_len=grid_dim_len, pix_arcsecs=pix_arcsecs, briggs_weight=briggs_weight)
    
    plt.imshow(dirty_psf[:,:,0])
    plt.savefig("output/dirty_psf_real.png")
    plt.imshow(dirty_psf[:,:,1])
    plt.savefig("output/dirty_psf_imag.png")
    
    plt.imshow(dirty_img[:,:,0])
    plt.savefig("output/dirty_img_real.png")
    plt.imshow(dirty_img[:,:,1])
    plt.savefig("output/dirty_img_imag.png")

    
