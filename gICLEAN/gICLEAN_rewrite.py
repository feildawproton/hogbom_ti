#gpu-login
#module load courses/cs205/2012

import numpy as np
import time,pdb,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
#import scikits.cuda.fft as fft
import skcuda.fft as fft
# Initialize the CUDA device
import pycuda.autoinit
# Elementwise stuff
from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath

from astropy.io import fits



######################
# CUDA kernels
######################

def cuda_compile(source_string, function_name):
  print("Compiling a CUDA kernel...")
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

GRID=lambda x,y,W: ((x)+((y)*W))

IGRIDX=lambda tid,W: tid%W
IGRIDY=lambda tid,W: int(tid)/int(W)

# -------------------
# Gridding kernels
# -------------------

code = \
"""
#define WIDTH 6
#define NCGF 12
#define HWIDTH 3
#define STEP 4

__device__ __constant__ float cgf[32];

// *********************
// MAP KERNELS
// *********************

__global__ void gridVis_wBM_kernel(float2 *Grd, float2 *bm, int *cnt, float *d_u, float *d_v, float *d_re, 
	float *d_im, int nu, float du, int gcount, int umax, int vmax, int2 *dbg_ndc){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  int u0 = 0.5*nu;
  if(iu >= u0 && iu <= u0+umax && iv <= u0+vmax){
    for (int ivis = 0; ivis < gcount; ivis++){
      float mu = d_u[ivis];
      float mv = d_v[ivis];
      int hflag = 1;
      if (mu < 0){
        hflag = -1;
        mu = -1*mu;
        mv = -1*mv;
      }
      float uu = mu/du+u0;
      float vv = mv/du+u0;
      int cnu=abs(iu-uu),cnv=abs(iv-vv);
      int ind = iv*nu+iu;
      if (cnu < HWIDTH && cnv < HWIDTH){
        //float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
        int cnu_ndx = int(round(4.6*cnu+NCGF-0.5));
        int cnv_ndx = int(round(4.6*cnv+NCGF-0.5));

        dbg_ndc[ind].x += cnu_ndx;
        dbg_ndc[ind].y += cnv_ndx;
        
        float wgt = cgf[cnu_ndx]*cgf[cnv_ndx];
        
        Grd[ind].x +=       wgt*d_re[ivis];
        Grd[ind].y += hflag*wgt*d_im[ivis];
        cnt[ind]   += 1;
        bm [ind].x += wgt;
       } 
      // deal with points&pixels close to u=0 boundary
      if (iu-u0 < HWIDTH && mu/du < HWIDTH) {
        mu = -1*mu;
        mv = -1*mv;
        uu = mu/du+u0;
        vv = mv/du+u0;
        cnu=abs(iu-uu),cnv=abs(iv-vv);
        if (cnu < HWIDTH && cnv < HWIDTH){
          //float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
          int cnu_ndx = int(round(4.6*cnu+NCGF-0.5));
          int cnv_ndx = int(round(4.6*cnv+NCGF-0.5));

          dbg_ndc[ind].x += cnu_ndx;
          dbg_ndc[ind].y += cnv_ndx;
          
          float wgt = cgf[cnu_ndx]*cgf[cnv_ndx];
          Grd[ind].x +=          wgt*d_re[ivis];
          Grd[ind].y += -1*hflag*wgt*d_im[ivis];
          cnt[ind]   += 1;
          bm [ind].x += wgt;
        }
      }
    }
  }
}

__global__ void dblGrid_kernel(float2 *Grd, int nu, int hfac){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  int u0 = 0.5*nu;
  if (iu > 0 && iu < u0 && iv < nu){
    int niu = nu-iu;   
    
    int niv = nu-iv;
    Grd[iv*nu+iu].x =      Grd[niv*nu+niu].x;
    Grd[iv*nu+iu].y = hfac*Grd[niv*nu+niu].y;
  }
}

__global__ void wgtGrid_kernel(float2 *Grd, int *cnt, float briggs, int nu){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  int u0 = 0.5*nu;
  if (iu >= u0 && iu < nu && iv < nu){
    if (cnt[iv*nu+iu]!= 0){
      int ind = iv*nu+iu;
      float foo = cnt[ind];
      float wgt = 1./sqrt(1 + foo*foo/(briggs*briggs));
      Grd[ind].x = Grd[ind].x*wgt;
      Grd[ind].y = Grd[ind].y*wgt;
    }
  }
}

__global__ void nrmGrid_kernel(float *Grd, float nrm, int nu){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  if (iu < nu && iv < nu){
      Grd[iv*nu + iu] = Grd[iv*nu+iu]*nrm;
  }
}

__global__ void corrGrid_kernel(float2 *Grd, float *corr, int nu){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  if (iu < nu && iv < nu){
      Grd[iv*nu + iu].x = Grd[iv*nu+iu].x*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
      Grd[iv*nu + iu].y = Grd[iv*nu+iu].y*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
  }
}

// *********************
// BEAM KERNELS
// *********************
__global__ void nrmBeam_kernel(float *bmR, float nrm, int nu){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  if(iu < nu && iv < nu){
    bmR[iv*nu+iu] = nrm*bmR[iv*nu+iu]; 
  }
}

// *********************
// MORE semi-USEFUL KERNELS
// *********************

__global__ void shiftGrid_kernel(float2 *Grd, float2 *nGrd, int nu){
  int iu = blockDim.x*blockIdx.x + threadIdx.x;
  int iv = blockDim.y*blockIdx.y + threadIdx.y;
  if(iu < nu && iv < nu){
    int niu,niv,nud2 = 0.5*nu;
    if(iu < nud2) 
      niu = nud2+iu;
    else 
      niu = iu-nud2;
    if(iv < nud2) 
      niv = nud2+iv;
    else 
      niv = iv-nud2;
    nGrd[niv*nu + niu].x = Grd[iv*nu+iu].x;
    nGrd[niv*nu + niu].y = Grd[iv*nu+iu].y;
  }
}

__global__ void trimIm_kernel(float2 *im, float *nim, int noff, int nx, int nnx){
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;
  if(iy < nnx && ix < nnx){
    nim[iy*nnx + ix] = im[(iy+noff)*nx+ix+noff].x;
  }
}
"""
module             = nvcc.SourceModule(code)
gridVis_wBM_kernel = module.get_function("gridVis_wBM_kernel") 
shiftGrid_kernel   = module.get_function("shiftGrid_kernel") 
nrmGrid_kernel     = module.get_function("nrmGrid_kernel") 
wgtGrid_kernel     = module.get_function("wgtGrid_kernel") 
dblGrid_kernel     = module.get_function("dblGrid_kernel") 
corrGrid_kernel    = module.get_function("corrGrid_kernel") 
nrmBeam_kernel     = module.get_function("nrmBeam_kernel") 
trimIm_kernel      = module.get_function("trimIm_kernel") 

# -------------------
# CLEAN kernels
# -------------------

find_max_kernel_source = \
"""
// Function to compute 1D array position
#define GRID(x,y,W) ((x)+((y)*W))

__global__ void find_max_kernel(float* dimg, int* maxid, float maxval, int W, int H, float* model)
{
  // Identify place on grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int id  = GRID(idy,idx,H);

  // Ignore boundary pixels
  if (idx>-1 && idx<W && idy>-1 && idy<H) {
    // Is this greater than the current max?
    if (dimg[id]==maxval) {
      // Do an atomic replace 
      // This might be #improvable#, but I think atomic operations are actually most efficient
      // in a situation like this where few (typically 1) threads will pass this conditional.
      // Note: this is a race condition!  If there are multiple instances of the max value, 
      // this will end up returning one randomly
      // See atomic operation info here: http://rpm.pbone.net/index.php3/stat/45/idpl/12463013/numer/3/nazwa/atomicExch
      // See also https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
      int dummy = atomicExch(maxid,id);
    }
  }
  // Update the model
  void __syncthreads();
  if (id==maxid[0]) {
    model[id]+=dimg[id];
  } 
}
"""
find_max_kernel = cuda_compile(find_max_kernel_source,"find_max_kernel")


sub_beam_kernel_source = \
"""
// Function to compute 1D array position
#define GRID(x,y,W) ((x)+((y)*W))
// Inverse
#define IGRIDX(x,W) ((x)%(W))
#define IGRIDY(x,W) ((x)/(W))

__global__ void sub_beam_kernel(float* dimg, float* dpsf, int* mid, float* cimg, float* cpsf, float scaler, int W, int H)
{
  // Identify place on grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int id  = GRID(idy,idx,H);
  // Identify position of maximum
  int midy = IGRIDX(mid[0],W);
  int midx = IGRIDY(mid[0],W);
  // Calculate position on the dirty beam
  int bidy = (idx-midx)+W/2;
  int bidx = (idy-midy)+H/2;
  int bid = GRID(bidx,bidy,W);

  // Stay within the bounds
  if (idx>-1 && idx<W && idy>-1 && idy<H && bidx>-1 && bidx<W && bidy>-1 && bidy<H) {
    // Subtract dirty beam from dirty map
    dimg[id]=dimg[id]-dpsf[bid]*scaler;
    // Add clean beam to clean map
    cimg[id]=cimg[id]+cpsf[bid]*scaler;
  };
}
"""
sub_beam_kernel = cuda_compile(sub_beam_kernel_source,"sub_beam_kernel")

add_noise_kernel = ElementwiseKernel(
"float *a, float* b, int N",
"b[i] = a[i]+b[i]",
"gpunoise")


######################
# Gridding functions
######################

def spheroid(eta,m,alpha):
  """
  Calculates spheriodal wave functions. See Schwab 1984 for details.  
  This implementation follows MIRIAD's grid.for subroutine.
  """

  twoalp = 2*alpha
  if np.abs(eta) > 1:
    print('bad eta value!')
  if (twoalp < 1 or twoalp > 4):
    print('bad alpha value!')
  if (m < 4 or m > 8):
    print('bad width value!')

  etalim = np.float32([1., 1., 0.75, 0.775, 0.775])
  nnum   = np.int8([5, 7, 5, 5, 6])
  ndenom = np.int8([3, 2, 3, 3, 3]) 
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

  q = np.float32(
        [ 
          [ [1., 9.077644E-1, 2.535284E-1],
                [1., 8.626056E-1, 2.291400E-1],
                [1., 8.212018E-1, 2.078043E-1],
                [1., 7.831755E-1, 1.890848E-1],
                [1., 7.481828E-1, 1.726085E-1]
          ],
          [ [1., 1.101270   , 3.858544E-1],
                [1., 1.025431   , 3.337648E-1],
                [1., 9.599102E-1, 2.918724E-1],
                [1., 9.025276E-1, 2.575337E-1],
                [1., 8.517470E-1, 2.289667E-1]
          ]
    ])

  i = int(m - 4)
  if(np.abs(eta) > etalim[i]):
    ip = 1
    x = eta*eta - 1
  else:
    ip = 0
    x = eta*eta - etalim[i]*etalim[i] 
  # numerator via Horner's rule 
  mnp  = nnum[i]-1
  num = p[ip,int(twoalp),mnp]
  for j in np.arange(mnp):
    num = num*x + p[ip,int(twoalp),mnp-1-j]
  # denominator via Horner's rule
  nq = ndenom[i]-1
  denom = q[ip,int(twoalp),nq]
  for j in np.arange(nq):
    denom = denom*x + q[ip,int(twoalp),nq-1-j]

  return  np.float32(num/denom)

def gcf(n,width):
  """
  Create table with spheroidal gridding function, C 
  This implementation follows MIRIAD's grid.for subroutine.
  """
  alpha = 1.
  j = 2*alpha
  p = 0.5*j
  phi = np.zeros(int(n),dtype=np.float32)
  for i in np.arange(n):
    x = np.float32(2*i-(n-1))/(n-1)
    phi[int(i)] = (np.sqrt(1-x*x)**j)*spheroid(x,width,p)
  return phi

def corrfun(n,width):
  """
  Create gridding correction function, c 
  This implementation follows MIRIAD's grid.for subroutine.
  """
  alpha = 1.
  dx = 2./n
  i0 = n/2+1
  phi = np.zeros(n,dtype=np.float32)
  for i in np.arange(n):
    x = (i-i0+1)*dx
    phi[i] = spheroid(x,width,alpha)
  return phi 

def cuda_gridvis(settings,plan):
  """
  Grid the visibilities parallelized by pixel.
  References:
    - Chapter 10 in "Interferometry and Synthesis in Radio Astronomy" 
        by Thompson, Moran, & Swenson
    - Daniel Brigg's PhD Thesis: http://www.aoc.nrao.edu/dissertations/dbriggs/ 
  """
  print("Gridding the visibilities")
  t_start=time.time()

  # unpack parameters
  vfile   = settings['vfile']
  briggs  = settings['briggs'] 
  imsize  = settings['imsize']
  cell    = settings['cell']
  nx      = np.int32(2*imsize)
  noff    = np.int32((nx-imsize)/2)

  ## constants
  arc2rad = np.float32(np.pi/180/3600.)
  du      = np.float32(1./(arc2rad*cell*nx))
  ## grab data
  f  = fits.open(settings['vfile'])
  ## quickly figure out what data is not flagged
  freq  = np.float32(f[0].header['CRVAL4'])
  good  = np.where(f[0].data.data[:,0,0,0,0,0,0] != 0)
  h_u   = np.float32(freq*f[0].data.par('uu')[good])
  h_v   = np.float32(freq*f[0].data.par('vv')[good])
  gcount = np.int32(np.size(h_u))
  ## assume data is unpolarized
  h_re   = np.float32(0.5*(f[0].data.data[good,0,0,0,0,0,0]+f[0].data.data[good,0,0,0,0,1,0]))
  h_im   = np.float32(0.5*(f[0].data.data[good,0,0,0,0,0,1]+f[0].data.data[good,0,0,0,0,1,1]))
  
  print("real", h_re.shape)
  print(h_re)
  print("imag", h_im.shape)
  print(h_im)
  print("u", h_u.shape)
  print(h_u)
  print("v", h_v.shape)
  print(h_v)
  
  ## make GPU arrays
  #h_grd  = np.zeros((nx,nx),dtype=np.complex64)
  #h_cnt  = np.zeros((nx,nx),dtype=np.int32)
  d_u    = gpu.to_gpu(h_u)
  d_v    = gpu.to_gpu(h_v)
  d_re   = gpu.to_gpu(h_re) 
  d_im   = gpu.to_gpu(h_im) 
  d_cnt  = gpu.zeros((nx,nx),np.int32)
  d_grd  = gpu.zeros((nx,nx),np.complex64)
  d_ngrd = gpu.zeros_like(d_grd)
  d_bm   = gpu.zeros_like(d_grd)
  d_nbm  = gpu.zeros_like(d_grd)
  d_fim  = gpu.zeros((imsize,imsize),np.float32)
  
  dbg_ndc = gpu.zeros((nx,nx,2), np.float32) 
  
  print("d_cnt", d_cnt.shape)
  print("d_grd", d_grd.shape)
  print("d_bm", d_bm.shape)
  
  ## define kernel parameters
  blocksize2D  = (8,16,1)
  gridsize2D   = (int(np.ceil(1.*nx/blocksize2D[0])),int(np.ceil(1.*nx/blocksize2D[1])))
  blocksizeF2D = (16,16,1)
  gridsizeF2D  = (int(np.ceil(1.*imsize/blocksizeF2D[0])),int(np.ceil(1.*imsize/blocksizeF2D[1])))
  #blocksize1D  = (256,1,1)
  #gridsize1D   = ((np.ceil(1.*gcount/blocksize1D[0])),1)

  # ------------------------
  # make gridding kernels
  # ------------------------
  ## make spheroidal convolution kernel (don't mess with these!)
  width = 6.
  ngcf  = 24.
  h_cgf = gcf(ngcf,width)
  ## make grid correction
  h_corr = corrfun(nx,width)
  
  d_cgf  = module.get_global('cgf')[0]
  print("what is d_cgf", d_cgf)
  d_corr = gpu.to_gpu(h_corr) 
  cu.memcpy_htod(d_cgf,h_cgf)
  print("what is h_cgf", h_cgf)

  # ------------------------
  # grid it up
  # ------------------------
  d_umax = gpu.max(cumath.fabs(d_u))
  d_vmax = gpu.max(cumath.fabs(d_v))
  umax   = np.int32(np.ceil(d_umax.get()/du))
  vmax   = np.int32(np.ceil(d_vmax.get()/du))

  ## grid ($$)
  #  This should be improvable via:
  #    - shared memory solution? I tried...
  #    - better coalesced memory access? I tried...
  #    - reorganzing and indexing UV data beforehand?
  #       (i.e. http://www.nvidia.com/docs/IO/47905/ECE757_Project_Report_Gregerson.pdf)
  #    - storing V(u,v) in texture memory?
  
  # 1.1)
  print("du", du)
  print("max and min of u", np.max(h_u), np.min(h_u), "with shape", h_u.shape)
  print("max and min of v", np.max(h_v), np.min(h_v), "with shape", h_v.shape)
  print("max and min of real", np.max(h_re), np.min(h_re), "with shape", h_re.shape)
  print("max and min of u", np.max(h_im), np.min(h_im), "with shape", h_im.shape)
  print("max and min of u", np.max(h_cgf), np.min(h_cgf), "with shape", h_cgf.shape)
  gridVis_wBM_kernel(d_grd,d_bm,d_cnt,d_u,d_v,d_re,d_im,nx,du,gcount,umax,vmax,dbg_ndc,\
			block=blocksize2D,grid=gridsize2D)
  
  #h_grd = d_grd.get()
  h_bm  = d_bm.get()
  '''
  plt.imshow(h_grd.real)
  plt.savefig("steps/1p1_half_grid_real.png")
  plt.imshow(h_grd.imag)
  plt.savefig("steps/1p1_half_grid_imag.png")
  '''
  
  plt.imshow(h_bm.real)
  plt.savefig("steps/1p1_half_bm_real.png")
  plt.imshow(h_bm.imag)
  plt.savefig("steps/1p1_half_bm_imag.png")
  
  print("max and min of real unweighted beam", np.max(h_bm.real), np.min(h_bm.real))
  print("max and min of imag unweighted beam", np.max(h_bm.imag), np.min(h_bm.imag))
  
  counts = d_cnt.get()
  print("max and min of counts", np.max(counts), np.average(counts), np.min(counts), "with shape", counts.shape, 
      "sum", np.sum(counts), "type", counts.dtype)
  
  
  cnt = d_cnt.get()
  plt.imshow(cnt)
  plt.savefig("steps/1p1_half_cnt.png")
  
  ndc = dbg_ndc.get()
  plt.imshow(ndc[:,:,0])
  plt.savefig("steps/1p1_half_dbg_ndc_x.png")
  plt.imshow(ndc[:,:,1])
  plt.savefig("steps/1p1_half_dbg_ndc_y.png")
  #print(ndc)
  

  # 1.2 ) 
  ## apply weights 
  wgtGrid_kernel(d_bm,d_cnt,briggs,nx,block=blocksize2D,grid=gridsize2D)
  
  # 1.3 ) 
  hfac = np.int32(1)
  dblGrid_kernel(d_bm,nx,hfac,block=blocksize2D,grid=gridsize2D)

  

  

  # 1.4 )
  shiftGrid_kernel(d_bm,d_nbm,nx,block=blocksize2D,grid=gridsize2D)
  
  h_bm  = d_nbm.get()

  plt.imshow(h_bm.real)
  plt.savefig("steps/1p4_shifted_bm_real.png")
  plt.imshow(h_bm.imag)
  plt.savefig("steps/1p4_shifted_bm_imag.png")
  
  print("max and min of real shifted beam", np.max(h_bm.real), np.min(h_bm.real))
  print("max and min of imag shifted beam", np.max(h_bm.imag), np.min(h_bm.imag))
  
  
  # 1.5) 
  ## normalize
  wgtGrid_kernel(d_grd,d_cnt,briggs,nx,block=blocksize2D,grid=gridsize2D)
  
  # 6.) 
  ## Reflect grid about v axis 
  hfac = np.int32(-1)
  dblGrid_kernel(d_grd,nx,hfac,block=blocksize2D,grid=gridsize2D)
  
  h_grd = d_grd.get()
  
  plt.imshow(h_grd.real)
  plt.savefig("steps/1p6_weighted_grid_real.png")
  plt.imshow(h_grd.imag)
  plt.savefig("steps/1p6_weights_grid_imag.png")
  
  
  # 1.7 ) 
  ## Shift both
  shiftGrid_kernel(d_grd,d_ngrd,nx,block=blocksize2D,grid=gridsize2D)
  
  '''
  h_grd = d_ngrd.get()
  plt.imshow(h_grd.real)
  plt.savefig("steps/1p7_shifted_grid_real.png")
  plt.imshow(h_grd.imag)
  plt.savefig("steps/1p7_shifted_grid_imag.png")
  '''
  
  # 2.) make bean
  # ------------------------
  # Make the beam
  # ------------------------
  ## Transform to image plane 
  # 2.1)  fft
  fft.fft(d_nbm,d_bm,plan) 
  '''
  print(type(plan))
  h_bm  = d_bm.get()
  plt.imshow(h_bm.real)
  plt.savefig("steps/2p1_fft_bm_real.png")
  plt.imshow(h_bm.imag)
  plt.savefig("steps/2p1_fft_bm_imag.png")
  
  h_bm  = d_nbm.get()
  plt.imshow(h_bm.real)
  plt.savefig("steps/2p1_fft_nbm_real.png")
  plt.imshow(h_bm.imag)
  plt.savefig("steps/2p1_fft_nbm_imag.png")
  '''
 
  ## Shift
  # 2.2)
  shiftGrid_kernel(d_bm,d_nbm,nx,block=blocksize2D,grid=gridsize2D)

  
  # 2.3)
  ## Correct for C
  corrGrid_kernel(d_nbm,d_corr,nx,block=blocksize2D,grid=gridsize2D)

  # Trim
  trimIm_kernel(d_nbm,d_fim,noff,nx,imsize,block=blocksizeF2D,grid=gridsizeF2D)
  
  #h_bm  = d_fim.get()
 
  
  ## Normalize
  d_bmax = gpu.max(d_fim)
  bmax = d_bmax.get()
  bmax = np.float32(1./bmax)
  nrmBeam_kernel(d_fim,bmax,imsize,block=blocksizeF2D,grid=gridsizeF2D)
  ## Pull onto CPU
  dpsf  = d_fim.get()


  # ------------------------
  # Make the map 
  # ------------------------
  # 2.6)
  ## Transform to image plane 
  fft.fft(d_ngrd,d_grd,plan)

  # 2.7)
  ## Shift
  shiftGrid_kernel(d_grd,d_ngrd,nx,block=blocksize2D,grid=gridsize2D)

  
  # 2.8)
  ## Correct for C
  corrGrid_kernel(d_ngrd,d_corr,nx,block=blocksize2D,grid=gridsize2D)
  

  
  # 2.9)
  ## Trim
  trimIm_kernel(d_ngrd,d_fim,noff,nx,imsize,block=blocksizeF2D,grid=gridsizeF2D)

  
  # 2.10)
  ## Normalize (Jy/beam)
  nrmGrid_kernel(d_fim,bmax,imsize,block=blocksizeF2D,grid=gridsizeF2D)
 

  ## Finish timers
  t_end=time.time()
  t_full=t_end-t_start
  print("Gridding execution time %0.5f"%t_full+' s')
  print("\t%0.5f"%(t_full/gcount)+' s per visibility')

  ## Return dirty psf (CPU) and dirty image (GPU)
  return dpsf,d_fim
  
######################
# CLEAN functions
######################

def serial_clean_beam(dpsf,window=20):
  """
  Clean a dirty beam on the CPU
  A very simple approach - just extract the central beam #improvable#
  Another solution would be fitting a 2D Gaussian, 
  e.g. http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
  """
  print("Cleaning the dirty beam")
  h,w=np.shape(dpsf)
  cpsf=np.zeros([h,w])
  cpsf[int(w/2-window):int(w/2+window),int(h/2-window):int(h/2+window)]=dpsf[int(w/2-window):int(w/2+window),int(h/2-window):int(h/2+window)]
  ##Normalize
  cpsf=cpsf/np.max(cpsf)  
  print("dirty beam max and min", np.max(cpsf), np.min(cpsf))
  return np.float32(cpsf)


def gpu_getmax(map):
  """
  Use pycuda to get the maximum absolute deviation of the residual map, 
  with the correct sign
  """
  imax=gpu.max(cumath.fabs(map)).get()
  if gpu.max(map).get()!=imax: imax*=-1
  return np.float32(imax)

def gpu_getmin(map):
  imin=gpu.min(cumath.fabs(map)).get()
  return imin

def cuda_hogbom(gpu_dirty,gpu_dpsf,gpu_cpsf,thresh=0.2,damp=1,gain=0.1,prefix='test'):
  """
  Use CUDA to implement the Hogbom CLEAN algorithm
  
  A nice description of the algorithm is given by the NRAO, here:
  http://www.cv.nrao.edu/~abridle/deconvol/node8.html
  
  Parameters:
  * dirty: The dirty image (2D numpy array)
  * dpsf: The dirty beam psf  (2D numpy array)
  * thresh: User-defined threshold to stop iteration, as a fraction of the max pixel intensity (float)
  * damp: The damping factor to scale the dirty beam by
  * prefix: prefix for output image file names
  """
  height,width=np.shape(gpu_dirty)
  ## Grid parameters - #improvable#
  tsize=8
  blocksize = (int(tsize),int(tsize),1)     	     # The number of threads per block (x,y,z)
  gridsize  = (int(width/tsize),int(height/tsize))   # The number of thread blocks     (x,y)
  ## Setup cleam image and point source model
  gpu_pmodel = gpu.zeros([height,width],dtype=np.float32)
  gpu_clean = gpu.zeros([height,width],dtype=np.float32)
  ## Setup GPU constants
  gpu_max_id = gpu.to_gpu(np.array([0]))
  imax=gpu_getmax(gpu_dirty)
  thresh_val=np.float32(thresh*imax)
  ## Steps 1-3 - Iterate until threshold has been reached
  t_start=time.time()
  i=0
  while abs(imax)>(thresh_val):
    if (np.mod(i,100)==0):
      print("Hogbom iteration",i)
    ## Step 1 - Find max
    find_max_kernel(gpu_dirty,gpu_max_id,imax,np.int32(width),np.int32(height),gpu_pmodel,\
			block=blocksize, grid=gridsize)
    ## Step 2 - Subtract the beam (assume that it is normalized to have max 1)
    ##          This kernel simultaneously reconstructs the CLEANed image.  
    dbg_min = gpu_getmin(gpu_dirty)
    print("Subtracting dirty beam "+str(i)+", maxval=%0.8f"%imax+' at x='+str(gpu_max_id.get()%width)+\
      ', y='+str(gpu_max_id.get()/width)+", curr min:", dbg_min)
    sub_beam_kernel(gpu_dirty,gpu_dpsf,gpu_max_id,gpu_clean,gpu_cpsf,np.float32(gain*imax),np.int32(width),\
			np.int32(height), block=blocksize, grid=gridsize)
    i+=1
    ## Step 3 - Find maximum value using gpuarray
    imax=gpu_getmax(gpu_dirty)
  t_end=time.time()
  t_full=t_end-t_start
  print( "Hogbom execution time %0.5f"%t_full+' s')
  print("\t%0.5f"%(t_full/i)+' s per iteration')
  ## Step 4 - Add the residuals back in
  add_noise_kernel(gpu_dirty,gpu_clean,width+height)
  return gpu_dirty,gpu_pmodel,gpu_clean

if __name__ == '__main__':
  
    
  example = 'mouse'
  ISIZE = 128
  CLEAN = True
  
  # Load settings for each example
  settings = dict([])
  if (example == 'gaussian'):
    # image a gaussian
    settings['vfile']  = 'examples/sim1.gauss.alma.out20.ms.fits'
    settings['imsize'] = np.int32(ISIZE)	# number of image pixels 
    settings['cell']   = np.float32(5.12/ISIZE)	# pixel size in arcseconds
    settings['briggs'] = np.float32(1e7)	# weight parameter
  elif (example == 'ring'):
    # image an inclined ring
    settings['vfile'] = 'examples/sim1.ring.alma.out20.ms.fits'
    settings['imsize']= np.int32(ISIZE)		# number of image pixels
    settings['cell']  = np.float32(5.12/ISIZE)	# pixel size in arcseconds
    settings['briggs']= np.float32(1e7)		# weight parameter
  elif (example == 'mouse'):
    # image a non-astronomical source
    settings['vfile'] = 'examples/sim1.mickey.alma.out20.ms.fits'
    settings['imsize']= np.int32(ISIZE)		# number of image pixels
    settings['cell']  = np.float32(5.12/ISIZE)	# pixel size in arcseconds
    settings['briggs']= np.float32(1e3)		# weight parameter
  elif (example == 'hd163296'):
    # image a single channel of the CO J=3-2 line from a protoplanetary disk
    # data from: https://almascience.nrao.edu/almadata/sciver/HD163296Band7/ 
    settings['vfile'] = 'examples/HD163296.CO32.regridded.ms.constub.c21.fits'
    settings['imsize']= np.int32(ISIZE)		# number of image pixels
    settings['cell']  = np.float32(25./ISIZE)	# pixel size in arcseconds
    settings['briggs']= np.float32(1e7)		# weight parameter
    vra = [-0.15,1.2]				# intensity range for figure
  else:
    print('QUITTING: NO SUCH EXAMPLE.')
    sys.exit()

  ## make cuFFT plan #improvable#
  imsize = settings['imsize']
  nx     = np.int32(2*imsize)
  plan   = fft.Plan((nx,nx),np.complex64,np.complex64) 

  ## Create the PSF & dirty image 
  dpsf,gpu_im = cuda_gridvis(settings,plan)
  
  gpu_dpsf = gpu.to_gpu(dpsf)
  
  #dirty = np.roll(np.fliplr(gpu_im.get()),1,axis=1) # why
  dirty = gpu_im.get()
  print("dirty image shape and type", dirty.shape, dirty.dtype)
  print("max and min of dirty image", np.max(dirty), np.min(dirty), dirty.dtype)
  
  plt.imshow(dirty.real)
  plt.savefig("output/dirty_img_real.png")
  plt.imshow(dirty.imag)
  plt.savefig("output/dirty_img_imag.png")
  
  plt.imshow(dpsf.real)
  plt.savefig("output/dirty_psf_real.png")
  plt.imshow(dpsf.imag)
  plt.savefig("output/dirty_psf_imag.png")

  np.save("output/dirty_img_real.npy", dirty.real)
  np.save("output/dirty_img_imag.npy", dirty.imag)
  np.save("output/dirty_psf_real.npy", dpsf.real)
  np.save("output/dirty_psf_imag.npy", dpsf.imag)
  
  ## Clean the PSF
    
  cpsf=serial_clean_beam(dpsf,imsize/50.)
  
  if CLEAN == True:

    gpu_cpsf = gpu.to_gpu(cpsf)
    

    print("Plotting dirty and cleaned beam")
    fig,axs=plt.subplots(1,2,sharex="row",sharey="col");plt.subplots_adjust(wspace=0)
    axs[0].imshow(dpsf,vmin=np.percentile(dpsf,1),vmax=np.percentile(dpsf,99),cmap=cm.gray)
    axs[1].imshow(cpsf,vmin=np.percentile(dpsf,1),vmax=np.percentile(dpsf,99),cmap=cm.gray)
    plt.savefig("output/test_cleanbeam.png")
    plt.close()
    
    ## Run CLEAN
    gpu_dirty,gpu_pmodel,gpu_clean = cuda_hogbom(gpu_im,gpu_dpsf,gpu_cpsf,thresh=0.2,gain=0.1)

    prefix=example
    try:
      vra
    except NameError:
      vra = [np.percentile(dirty,1),np.percentile(dirty,99)]

    print("Plotting dirty image and dirty image after iterative source removal")
    fig,axs=plt.subplots(1,2,sharex="row",sharey="col",figsize=(12.2,6));plt.subplots_adjust(wspace=0)
    axs[0].imshow(dirty,vmin=vra[0],vmax=vra[1],cmap=cm.gray,origin='lower')
    axs[0].set_title('Original dirty image')
    axs[1].imshow(np.roll(np.fliplr(gpu_dirty.get()),1,axis=1),vmin=vra[0],vmax=vra[1],cmap=cm.gray,origin='lower')
    axs[1].set_title('Dirty image cleaned of sources')
    plt.savefig("output/"+prefix+'_dirty_final.png')
    plt.close()
    
    print("Plotting dirty image and final clean image")
    vra = [np.percentile(dirty,1),np.percentile(dirty,99)]
    fig,axs=plt.subplots(1,2,sharex="row",sharey="col",figsize=(12.2,6));plt.subplots_adjust(wspace=0)
    clean = np.roll(np.fliplr(gpu_clean.get()),1,axis=1)
    axs[0].imshow(dirty,vmin=vra[0],vmax=vra[1],cmap=cm.gray,origin='lower')
    axs[0].set_title('Original dirty image')
    axs[1].imshow(clean,vmin=vra[0],vmax=vra[1],cmap=cm.gray,origin='lower')
    axs[1].set_title('Final cleaned image')
    plt.savefig("output/"+prefix+'_clean_final.png')
    plt.close()
    
