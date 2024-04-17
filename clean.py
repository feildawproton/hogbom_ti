# -- GLOBAL IMPORTS --
import numpy as np
from typing import Tuple
import taichi as ti


def clean_beam_by_window(dirty_psf: np.ndarray, window=20):
    # a better solution would be to fit a gaussian according the gICLEAN
    # they suggest this: http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
    
    print("clean beam indexing needs a revisit.  but this is temporary anyway")
    
    clean_beam = np.zeros(shape=dirty_psf.shape)
    
    width   = dirty_psf.shape[0]
    height  = dirty_psf.shape[1]
    '''
    x_start = int(width/2 - window) - 1 # in case to small
    x_stop  = int(width/2 + window) + 3  # come back to this
    y_start = int(height/2 - window) - 1
    y_stop  = int(height/2 + window) + 3
    print(x_start, x_stop, y_start, y_stop)
    clean_beam[x_start:x_stop,y_start:y_stop,:] = dirty_psf[x_start:x_stop,y_start:y_stop,:]
    
    #clean_beam[width/2, height/2] 
    
    #max_cmplx  = np.max(clean_beam, axis=(0,1))
    max_real = np.max(clean_beam[:,:,0])
    clean_beam[:,:,0] = np.multiply(clean_beam[:,:,0], 1/max_real)
    '''
    
    clean_beam[int(width/2),int(height/2),0] = 1.0
    
    return clean_beam


def run_hogbom(dirty_img: np.ndarray, dirty_psf: np.ndarray, clean_psf: np.ndarray, thresh=0.2, damp=1., gain=0.1
              ) -> Tuple[np.ndarray, np.ndarray]:
    # http://www.cv.nrao.edu/~abridle/deconvol/node8.html
    # parameters
    # dirty image
    # dirty beam point spread function
    # fraction of max pixel intensity threshold
    # damping factor to scale dirty beam

    dirty_img_cpy = np.copy(dirty_img)
    clean_img     = np.zeros(shape=dirty_img_cpy.shape, dtype=np.float32)
    
    max_col_ndc = []
    max_row_ndc = []
    
    for iter in range(10):
        abs_real = np.abs(dirty_img_cpy[:,:,0])
        max_ndx  = np.argmax(abs_real)
        col_ndx  = int(max_ndx/abs_real.shape[0])
        row_ndx  = int(max_ndx % abs_real.shape[0])
        
        x_max_ndx = col_ndx
        y_max_ndx = row_ndx
        max_col_ndc.append(col_ndx)
        max_row_ndc.append(row_ndx)
        
        x_offset = int(x_max_ndx - dirty_psf.shape[0]/2)
        
        view_x_start = x_offset
        view_x_start = max(view_x_start, 0)
        view_x_stop  = dirty_psf.shape[0] + x_offset
        view_x_stop  = min(dirty_psf.shape[0], view_x_stop)
        
        srce_x_start = -x_offset
        srce_x_start = max(srce_x_start, 0)
        srce_x_stop  = dirty_psf.shape[0] - x_offset
        srce_x_stop  = min(dirty_psf.shape[0], srce_x_stop)
        
        y_offset = int(y_max_ndx - dirty_psf.shape[1]/2)
        
        view_y_start = y_offset
        view_y_start = max(view_y_start, 0)
        view_y_stop  = dirty_psf.shape[1] + y_offset
        view_y_stop  = min(dirty_psf.shape[1], view_y_stop)
        
        srce_y_start = -y_offset
        srce_y_start = max(srce_y_start, 0)
        srce_y_stop  = dirty_psf.shape[1] - y_offset
        srce_y_stop  = min(dirty_psf.shape[1], srce_y_stop)
        
        #print("found offsets:", x_offset, y_offset)
        #print("view will start at", view_x_start, view_y_start, "and go to", view_x_stop, view_y_stop, "with spans", view_x_stop-view_x_start, view_y_stop-view_y_start)
        #print("source will start at", srce_x_start, srce_y_start, "and go to", srce_x_stop, srce_y_stop, "with spans", srce_x_stop-srce_x_start, srce_y_stop-srce_y_start)

        
        offset_dirty_psf = np.zeros(shape=dirty_psf.shape)
        offset_dirty_psf[view_x_start:view_x_stop,view_y_start:view_y_stop] = dirty_psf[srce_x_start:srce_x_stop,srce_y_start:srce_y_stop]
        
        offset_clean_psf = np.zeros(shape=clean_psf.shape)
        offset_clean_psf[view_x_start:view_x_stop,view_y_start:view_y_stop] = clean_psf[srce_x_start:srce_x_stop,srce_y_start:srce_y_stop]
        
        max_real      = dirty_img_cpy[x_max_ndx, y_max_ndx, 0]
        beam_scale    = max_real*gain
        print("max abs real", abs_real[x_max_ndx, y_max_ndx], "and real val there", max_real)
        print("found at", x_max_ndx, y_max_ndx, "with scale", beam_scale)
        print("current min real:", np.min(dirty_img_cpy[:,:,0]))
        dirty_scaled  = np.multiply(offset_dirty_psf, beam_scale)
        dirty_img_cpy = np.subtract(dirty_img_cpy, dirty_scaled)
        
        #print("shape of dirty img copy", dirty_img_cpy.shape)
        
        clean_scaled  = np.multiply(offset_clean_psf, beam_scale)
        clean_img     = np.add(clean_img, clean_scaled)
        
    
    '''
    plt.imshow(dirty_img_cpy[:,:,0])
    #plt.scatter(x_max_np, y_max_np, s=20, c="red", marker='x')
    plt.scatter(row_ndx, col_ndx, s=20, c="red", marker='x')
    plt.savefig("steps/3p2_dirt_img_1_iter_real.png")
    plt.imshow(dirty_img_cpy[:,:,1])
    plt.savefig("steps/3p2_dirt_img_1_iter_imag.png")
    ''' 

    
    plt.imshow(offset_dirty_psf[:,:,0])
    plt.savefig("steps/3p2_offset_dirt_psf_real.png")
    plt.imshow(offset_dirty_psf[:,:,1])
    plt.savefig("steps/3p2_offset_dirt_psf_imag.png")
    
    plt.imshow(offset_clean_psf[:,:,0])
    plt.savefig("steps/3p2_offset_clean_psf_real.png")
    plt.imshow(offset_clean_psf[:,:,1])
    plt.savefig("steps/3p2_offset_clean_psf_imag.png")
    
    plt.imshow(dirty_img_cpy[:,:,0])
    #plt.scatter(row_ndx, col_ndx, s=20, c="red", marker='x')
    plt.savefig("steps/3p2_dirt_img_1_iter_real.png")
    plt.imshow(dirty_img_cpy[:,:,1])
    plt.savefig("steps/3p2_dirt_img_1_iter_imag.png")
    
    plt.imshow(clean_img[:,:,0])
    #plt.scatter(x_max_np, y_max_np, s=20, c="red", marker='x')
    #plt.scatter(row_ndx, col_ndx, s=20, c="red", marker='x')
    plt.savefig("steps/3p2_clean_img_1_iter_real.png")
    plt.imshow(clean_img[:,:,1])
    plt.savefig("steps/3p2_clean_img_1_iter_imag.png")
    
    print("test indexing")
    dirty_img[col_ndx, row_ndx, :] = 1
    plt.imshow(dirty_img_cpy[:,:,0])
    plt.scatter(max_row_ndc, max_col_ndc, s=20, c="red", marker='x')
    plt.savefig("steps/3p2_dirt_img_test_indexing_real.png")
    plt.imshow(dirty_img_cpy[:,:,1])
    plt.savefig("steps/3p2_dirt_img_test_indexing_imag.png")

#psf is the same as beam
def run_clean(dirty_psf: np.ndarray, dirty_img: np.ndarray): 
    '''  
    min_img_real = np.min(dirty_img[:,:,0])
    dirty_img[:,:,0] = np.subtract(dirty_img[:,:,0], min_img_real)
    max_img_real = np.max(dirty_img[:,:,0])
    dirty_img[:,:,0] = np.add(dirty_img[:,:,0], max_img_real) 
    
    min_psf_real = np.min(dirty_psf[:,:,0])
    dirty_psf[:,:,0] = np.add(dirty_psf[:,:,0], min_psf_real)
    max_psf_real = np.max(dirty_psf[:,:,0])
    dirty_psf[:,:,0] = np.add(dirty_img[:,:,0], max_psf_real) 
    
    print("dirty beam real max and min", np.max(dirty_psf[:,:,0]), np.min(dirty_psf[:,:,0]))
    print("dirty image real max and min", np.max(dirty_img[:,:,0]), np.min(dirty_img[:,:,0]))
    '''
    
    # 1.) -- CLEAN THE PSF --
    clean_psf = clean_beam_by_window(dirty_psf=dirty_psf, window=dirty_img.shape[0]/50)
    
    plt.imshow(clean_psf[:,:,0])
    plt.savefig("output/clean_psf_real.png")
    plt.imshow(clean_psf[:,:,1])
    plt.savefig("output/clean_psf_imag.png")
    
    run_hogbom(dirty_img=dirty_img, dirty_psf=dirty_psf, clean_psf=clean_psf)
    
if __name__ == "__main__":
    # -- GLOBAL FOR TESTING --
    from matplotlib import pyplot as plt
    
    # -- LOCAL FOR TESTING
    from fits_handler import FitsFile    
    
    fits_file = FitsFile(path="demo_data/sim1.mickey.alma.out20.ms.fits", use_ti=True)
    grid_dim_len  = 64 # characteristic size of the image, size will be grid_dim_len*grid_dim_len
    pix_arcsecs   = 5.12 / grid_dim_len
    briggs_weight = 1e3
    dirty_psf, dirty_img = fits_file.convert_to_grid(grid_dim_len=grid_dim_len, pix_arcsecs=pix_arcsecs, briggs_weight=briggs_weight)
    
    print("dirty beam real max and min", np.max(dirty_psf[:,:,0]), np.min(dirty_psf[:,:,0]))
    print("dirty image real max and min", np.max(dirty_img[:,:,0]), np.min(dirty_img[:,:,0]))
    
    plt.imshow(dirty_psf[:,:,0])
    plt.savefig("output/dirty_psf_real.png")
    plt.imshow(dirty_psf[:,:,1])
    plt.savefig("output/dirty_psf_imag.png")
    
    plt.imshow(dirty_img[:,:,0])
    plt.savefig("output/dirty_img_real.png")
    plt.imshow(dirty_img[:,:,1])
    plt.savefig("output/dirty_img_imag.png")
    
    
    run_clean(dirty_psf=dirty_psf, dirty_img=dirty_img)
    

    
    
    
    