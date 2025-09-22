#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void replicate0(int tot_size, char* flags_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x < tot_size) {
        flags_d[gid_x] = 0;
    }
}

__global__ void mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    // Mat rows must be equal to the size of the mat_shp_sc_d array
    // Length is equal to the size of flags_d since it is based on tot_size
    
    if (gid_x < mat_rows) {
        if (gid_x == 0) {
            flags_d[0] = 1;
        }
        else {
            const unsigned int flag = mat_shp_sc_d[gid_x-1];
            flags_d[flag] = 1;
        }
    }
}


__global__ void mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x < tot_size) {
        const unsigned int i = mat_inds[gid_x];
        const float v = mat_vals[gid_x];
        // We use the temorary array since the same index are accessed concurrently in the vector
        tmp_pairs[gid_x] = vct[i] * v; 
    }
}


__global__ void select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x < mat_rows) {
        const unsigned int i = mat_shp_sc_d[gid_x];
        res_vct_d[gid_x] = tmp_scan[i-1];
    }
}

#endif // SPMV_MUL_KERNELS
