#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS


__global__ void replicate0(int tot_size, char* flags_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x < tot_size) {
        flags_d[gid_x] = 0;
    }
}

// let mkFlagArray 't [m] 
//             (aoa_shp: [m]i64) (zero: t)   --aoa_shp=[0,3,1,0,4,2,0]
//             (aoa_val: [m]t  ) : []t   =   --aoa_val=[1,1,1,1,1,1,1]
//   let shp_rot = map (\i->if i==0 then 0   --shp_rot=[0,0,3,1,0,4,2]
//                          else aoa_shp[i-1]
//                     ) (iota m)
//   let shp_scn = scan (+) 0 shp_rot       --shp_scn=[0,0,3,4,4,8,10]
//   let aoa_len = if m == 0 then 0         --aoa_len= 10
//                 else shp_scn[m-1]+aoa_shp[m-1]
//   let shp_ind = map2 (\shp ind ->        --shp_ind= 
//                        if shp==0 then -1 --  [-1,0,3,-1,4,8,-1]
//                        else ind          --scatter
//                      ) aoa_shp shp_scn   --   [0,0,0,0,0,0,0,0,0,0]
//   in scatter (replicate aoa_len zero)    --   [-1,0,3,-1,4,8,-1]
//              shp_ind aoa_val             --   [1,1,1,1,1,1,1]
//                                      -- res = [1,0,0,1,1,0,0,0,1,0] 


__global__ void mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    // Mat rows must be equal to the size of the mat_shp_sc_d array

    // Exlusive scan 
    if (gid_x > 1) {
        const unsigned int flag = mat_shp_sc_d[gid_x-1] + mat_shp_sc_d[gid_x];
        if (flag < mat_rows) {
            flags_d[mat_shp_sc_d[gid_x]] = 1;
        }  
        // mat_shp_sc_d[gid_x] = mat_shp_sc_d[gid_x-1] + mat_shp_sc_d[gid_x] ;
    } else {
        flags_d[0] = 1; // Maybe check for zero elements in flags_d
    }

    // Length is equal to the size of flags_d since it is based on tot_size
    // if (mat_shp_sc_d[gid_x] < mat_rows) {
    //     flags_d[mat_shp_sc_d[gid_x]] = 1;
    // }
}


__global__ void mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // map (\(i, v) -> vct[i] * v) mat_val 

    if (gid_x < tot_size) {
        const unsigned int i = mat_inds[gid_x];
        const unsigned int v = mat_vals[gid_x];
        // We use the temorary array since the same index are accessed concurrently in the vector
        tmp_pairs[gid_x] = vct[i] * v ; 
    }
}


__global__ void select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // map (\i -> tmp_scan[i-1]) shp_sc
    
    if (gid_x < mat_rows) {
        const unsigned int i = mat_shp_sc_d[gid_x];
        res_vct_d[gid_x] = tmp_scan[i-1];
    }

}

#endif // SPMV_MUL_KERNELS
