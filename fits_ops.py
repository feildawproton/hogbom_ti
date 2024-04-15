# -- GLOBAL IMPORTS -- 
from astropy.io import fits
import numpy as np
import copy
import math
import taichi as ti
import scikits.cuda.fft as fft

# -- GLOBA FOR TESTING --
from matplotlib import pyplot as plt

# -- PLEASE TREAT AS CONST --
ARC_TO_RAD = math.pi / 180. / 3600.
HWIDTH     = 3
NCGF       = 12


@ti.kernel
def grid_BM_mirrored(n_good:int, du: float, u: ti.template(), v: ti.template(), bm: ti.template()):
    nu = bm.shape[0]
    u0 = int(nu * 0.5)
    for x_ndx, y_ndx in bm:
        u_ndx = y_ndx
        v_ndx = x_ndx
        if u_ndx >= u0: #and u_ndx <= u0+u_max and v_ndx <= u0+v_max:
            u_ndx = y_ndx
            v_ndx = x_ndx
        else:
            u_ndx = nu - y_ndx
            v_ndx = nu - x_ndx

        for vis_ndx in range(n_good):
            mu = u[vis_ndx]
            mv = v[vis_ndx]
            
            if mu < 0:
                mu = -1*mu
                mv = -1*mv
            
            uu  = mu/du+u0
            vv  = mv/du+u0
            cnu = ti.abs(u_ndx-uu)
            cnv = ti.abs(v_ndx-vv)
            if cnu < HWIDTH and cnv < HWIDTH:
                #wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
                wgt = ((4.6*cnu+NCGF-0.5))*((4.6*cnv+NCGF-0.5))
                bm[x_ndx, y_ndx][0] += wgt
            
            if (u_ndx-u0) < HWIDTH and (mu/du) < HWIDTH:
                mu  = -1*mu
                mv  = -1*mv
                uu  = mu/du+u0
                vv  = mv/du+u0
                cnu = ti.abs(u_ndx-uu)
                cnv = ti.abs(v_ndx-vv)
                if cnu < HWIDTH and cnv < HWIDTH:
                    wgt = ((4.6*cnu+NCGF-0.5))*((4.6*cnv+NCGF-0.5))
                    bm[x_ndx, y_ndx][0] += wgt
                
                    

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
def mirror_u(grid: ti.template(), half_u_grid: ti.template(), imag_fac: float):
    print("check that we were supposed to mirror on u only and not flip v like the reference")
    u_len = grid.shape[0]
    half_u_len = half_u_grid.shape[0]
    for u_ndx, v_ndx in grid:
        u_h_ndx = u_ndx
        #h_v_ndx = u_len - v_ndx
        h_v_ndx = v_ndx
        if u_ndx >= half_u_len:
            u_h_ndx = u_ndx - half_u_len
        else:
            u_h_ndx = half_u_len - u_ndx
        grid[u_ndx, v_ndx][0] = half_u_grid[u_h_ndx, h_v_ndx][0]
        grid[u_ndx, v_ndx][1] = half_u_grid[u_h_ndx, h_v_ndx][1] * imag_fac


@ti.kernel
def double_grid(half_u_grid: ti.template(), imag_fac: float, grid: ti.template()):
    nu = grid.shape[1]
    u0 = half_u_grid.shape[0]
    for u_ndx, v_ndx in half_u_grid:
        grid[u_ndx+u0, v_ndx][0] = half_u_grid[u_ndx, v_ndx][0]
        grid[u_ndx+u0, v_ndx][1] = half_u_grid[u_ndx, v_ndx][1] * imag_fac
    
    for u_ndx, v_ndx in grid:
        if u_ndx < u0:
            u_h_ndx = nu - u_ndx
            v_h_ndx = nu - v_ndx
            grid[u_ndx, v_ndx][0] = grid[u_h_ndx, v_h_ndx][0]
            grid[u_ndx, v_ndx][1] = grid[u_h_ndx, v_h_ndx][1] * imag_fac
    

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
        self.hdu_list.close()
        
    @ti.kernel
    def __fill_reduced_grid(self):                
        for x_ndx, y_ndx in self.grid:
            for f_ndx in range(0, self.n_good):
                mu = self.u[f_ndx]
                mv = self.v[f_ndx]
                
                hflag = 1.0
                if mu < 0:
                    hflag = -1.
                    mu = -1.*mu
                    mv = -1.*mv
                    
                vv  = mv/self.du+self.u_start
                cnu = ti.abs(x_ndx-(mu/self.du))
                cnv = ti.abs(y_ndx-vv)
                
                # -- FILL IN --
                if cnu < HWIDTH and cnv < HWIDTH:
                    wgt = (4.6*cnu+NCGF-0.5)*(4.6*cnv+NCGF-0.5)
                    self.grid[x_ndx, y_ndx][0] +=       wgt*self.real[f_ndx]
                    self.grid[x_ndx, y_ndx][1] += hflag*wgt*self.imag[f_ndx]
                    self.cnts[x_ndx, y_ndx]    += 1
                    self.bm[x_ndx, y_ndx][0]   += wgt
                # -- POINTS AND PIXELS NEAR u=0 NEED EXTRA CARE
                if (x_ndx) < HWIDTH and (mu/self.du) < HWIDTH:
                    mu = -1.*mu
                    mv = -1.*mv
                    vv  = mv/self.du+self.u_start
                    cnu = ti.abs(x_ndx-(mu/self.du))
                    cnv = ti.abs(y_ndx-vv)
                    if cnu < HWIDTH and cnv < HWIDTH:
                        wgt = (4.6*cnu+NCGF-0.5)*(4.6*cnv+NCGF-0.5)
                        self.grid[x_ndx, y_ndx][0] +=           wgt*self.real[f_ndx]
                        self.grid[x_ndx, y_ndx][1] += -1.*hflag*wgt*self.imag[f_ndx]
                        self.cnts[x_ndx, y_ndx]    += 1
                        self.bm[x_ndx, y_ndx][0]   += wgt
                        
    def __make_grid_py(self):
        u_start = int(0.5 * self.nx)        #u0
        u_stop  = u_start + self.u_max
        v_stop  = u_start + self.v_max
        # -- ACROSS THE WIDTH, U DIMENSION
        for u_ndx in range(max(u_start, 0), min(u_stop, self.grid.shape[0])):
                # -- ACROSS THE HEIGHT, V DIMENSION
            for v_ndx in range(0, min(self.grid.shape[1], v_stop)):
                for vis_ndx in range(0, self.n_good):
                    mu = self.u[vis_ndx]
                    mv = self.v[vis_ndx]
                    
                    if mu < 0:
                        hflag = -1.
                        mu = -1.*mu
                        mv = -1.*mv
                    else:
                        hflag = 1.
                        
                    uu  = mu/self.du+u_start
                    vv  = mv/self.du+u_start
                    cnu = abs(u_ndx-uu)
                    cnv = abs(v_ndx-vv)
                    #ndx = v_ndx*self.nu+u_ndx
                    
                    # -- FILL IN --
                    if cnu < HWIDTH and cnv < HWIDTH:
                        wgt = round(4.6*cnu+NCGF-0.5)*round(4.6*cnv+NCGF-0.5)
                        self.grid.real[u_ndx, v_ndx] +=       wgt*self.real[vis_ndx]
                        self.grid.imag[u_ndx, v_ndx] += hflag*wgt*self.imag[vis_ndx]
                        self.cnts[u_ndx, v_ndx]      += 1
                        self.bm.real[ u_ndx, v_ndx]        += wgt
                    # -- POINTS AND PIXELS NEAT u=0 NEED EXTRA CARE
                    if (u_ndx-u_start) < HWIDTH and (mu/self.du) < HWIDTH:
                        mu = -1.*mu
                        mv = -1.*mv
                        uu = mu/self.du+u_start
                        vv = mv/self.du+u_start
                        cnu = abs(u_ndx-uu)
                        cnv = abs(v_ndx-vv)
                        if cnu < HWIDTH and cnv < HWIDTH:
                            wgt = round(4.6*cnu+NCGF-0.5)*round(4.6*cnv+NCGF-0.5)
                            self.grid.real[u_ndx, v_ndx]  +=           wgt*self.real[vis_ndx]
                            self.grid.imag[u_ndx, v_ndx]  += -1.*hflag*wgt*self.imag[vis_ndx]
                            self.cnts[u_ndx, v_ndx]       += 1
                            self.bm.real[u_ndx, v_ndx]    += wgt

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

            self.nx          = 2*grid_dim_len
            self.du          = 1. / (ARC_TO_RAD * pix_arcsecs * self.nx)  
            self.briggs_wght = briggs_weight          

            data     = self.hdu_list[0].data
            good_ndc = np.where(data.data[:,0,0,0,0,0,0] != 0)[0]
            good_u   = data.par("uu")[good_ndc]
            good_v   = data.par("vv")[good_ndc]
            
            self.n_good = good_ndc.shape[0]            
            
            real = np.add(data.data[good_ndc,0,0,0,0,0,0], data.data[good_ndc,0,0,0,0,1,0])
            real = np.multiply(0.5, real)
            imag = np.add(data.data[good_ndc,0,0,0,0,0,1], data.data[good_ndc,0,0,0,0,1,1])
            imag = np.multiply(0.5, imag)
            
            freq = self.hdu_list[0].header['CRVAL4']
            u    = np.multiply(freq, good_u)
            v    = np.multiply(freq, good_v)
            
            self.u_start = int(0.5 * self.nx)
            
            u_max_abs  = np.max(np.fabs(u))
            v_max_abs  = np.max(np.fabs(v))
            u_max      = int(np.ceil(u_max_abs/self.du))
            v_max      = int(np.ceil(v_max_abs/self.du))
            u_stop     = self.u_start + u_max
            u_stop     = min(u_stop, self.nx)
            v_stop     = self.u_start + v_max
            v_stop     = min(v_stop, self.nx)
            x_red_size = u_stop-self.u_start
            y_red_size = v_stop
            
            print("real", real.shape)
            print(real)
            print("imag", imag.shape)
            print(imag)
            print("u", u.shape)
            print(u)
            print("v", v.shape)
            print(v)
            print("du", self.du)
            print("u_max", u_max)
            print("v_max", v_max)
            

            
            if self.use_ti == True:
                self.real = ti.field(dtype=ti.f32, shape=real.shape)
                self.real.from_numpy(real)
                self.imag = ti.field(dtype=ti.f32, shape=imag.shape)
                self.imag.from_numpy(imag)
                
                self.u = ti.field(dtype=ti.f32, shape=u.shape)
                self.u.from_numpy(u)
                self.v = ti.field(dtype=ti.f32, shape=v.shape)
                self.v.from_numpy(v)
                
                self.grid = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                self.cnts = ti.field(shape=(u_stop, v_stop), dtype=ti.int32)
                self.bm   = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                
                print("cnt shape", self.cnts.shape)
                print("grid shape", self.grid.shape)
                print("bm shape", self.bm.shape)
                
                
                # 1.) GRID
                # 1.1) fill in the reduced grids
                grid_BM_mirrored(u_max=u_max, v_max=v_max, n_good=self.n_good, du=self.du, u=self.u, v=self.v, bm=self.bm)

                h_bm = self.bm.to_numpy()
                plt.imshow(h_bm[:,:,0])
                plt.savefig("steps/1p1_half_bm_real.png")
                plt.imshow(h_bm[:,:,1])
                plt.savefig("steps/1p1_half_bm_imag.png")
                '''
                print("plotting weighted bm")
                h_grd = self.grid.to_numpy()
                plt.imshow(h_grd[:,:,0])
                plt.savefig("steps/1p1_half_grid_real.png")
                plt.imshow(h_grd[:,:,1])
                plt.savefig("steps/1p1_half_grid_imag.png")
                '''
                
                keep_going = False
                
                if keep_going:


                    # 1.2 ) apply weights to bm
                    apply_weights(count=self.half_cnts, briggs_wght=self.briggs_wght, grid=self.half_bm)
                    '''
                    print("plotting weighted bm")
                    grid = self.half_bm.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''

                    # 1.3 ) mirror bm u (flipping v)
                    self.bm = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                    double_grid(half_u_grid=self.half_bm, imag_fac=1., grid=self.bm)
                    
                    '''
                    print("plotting doubled bm")
                    grid = self.bm.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''
                    
                    # 1.4 ) shift_bm
                    self.shift_bm = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                    shift(self.bm, self.shift_bm)

                    '''
                    print("plotting shifted bm")
                    grid = self.shift_bm.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''
                    
                    # 1.5 ) weight grid
                    apply_weights(count=self.half_cnts, briggs_wght=self.briggs_wght, grid=self.half_grid)
                    
                    '''
                    print("plotting weighted grid")
                    grid = self.half_grid.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''
                    
                    # 1.6 ) reflect grid
                    self.grid = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                    double_grid(half_u_grid=self.half_grid, imag_fac=-1, grid=self.grid)
                    
                    '''
                    print("plotting doubled grid")
                    grid = self.grid.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''
                    
                    # 1.7 ) shift grid
                    self.shift_grid = ti.Vector.field(n=2, shape=(u_stop, v_stop), dtype=ti.f32)
                    shift(self.grid, self.shift_grid)
                    '''
                    print("plotting shifted grid")
                    grid = self.shift_grid.to_numpy()
                    plt.imshow(grid[:,:,0])
                    plt.show()
                    plt.imshow(grid[:,:,1])
                    plt.show()
                    '''
                    # 2.) MAKE BEAM
                    #self.plan  = fft.Plan((self.nx,self.nx),np.complex64,np.complex64)
                    #fft.fft(self.shift_bm, self.bm, self.plan)
                    
                    shift_bm_np = self.shift_bm.to_numpy()
                    
                    print(shift_bm_np.dtype)
                    
                    fft_bm = np.fft.fft2(a=shift_bm_np)
                    
                    print("plotting fft beam")
                    grid = fft_bm
                    plt.imshow(grid.real[:,:,0])
                    plt.show()
                    plt.imshow(grid.real[:,:,1])
                    plt.show()
                    plt.imshow(grid.imag[:,:,0])
                    plt.show()
                    plt.imshow(grid.imag[:,:,1])
                    plt.show()

                
                
            
            return 
            
if __name__ == "__main__":
    
    
    fits_file = FitsFile(path="demo_data/sim1.mickey.alma.out20.ms.fits", use_ti=True)
    grid_dim_len = 64 # characteristic size of the image, size will be grid_dim_len*grid_dim_len
    pix_arcsecs  = 5.12 / grid_dim_len
    briggs_weight = 1e3
    _ = fits_file.convert_to_grid(grid_dim_len=grid_dim_len, pix_arcsecs=pix_arcsecs, briggs_weight=briggs_weight)
    
    #plt.imshow(image.real)
    #plt.show()
    #plt.imshow(image.imag)
    #plt.show()
    
    
