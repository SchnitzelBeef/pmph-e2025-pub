-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let exscan f ne xs =
  map2 (\i x -> if i == 0 then ne else x)
       (indices xs)
       (rotate (-1) (scan f ne xs))

-- From spMVmult-flat.fut (weeklies-1):
let mkFlagArray 't [m] 
            (aoa_shp: [m]i64) (zero: t)   --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t  ) : []t   =   --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0   --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot       --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = if m == 0 then 0         --aoa_len= 10
                else shp_scn[m-1]+aoa_shp[m-1]
  let shp_ind = map2 (\shp ind ->        --shp_ind= 
                       if shp==0 then -1 --  [-1,0,3,-1,4,8,-1]
                       else ind          --scatter
                     ) aoa_shp shp_scn   --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate aoa_len zero)    --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val             --   [1,1,1,1,1,1,1]
                                     -- res = [1,0,0,1,1,0,0,0,1,0] 

-- From spMVmult-flat.fut (weeklies-1):
let sgmScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (flags: [n]i64)
            (vals: [n]t)
            : [n]t =
  scan (\(f1, v1) (f2, v2) -> (if f1 == 0 || f2 == 0 then 0 else 1, if f2 != 0 then v2 else op v1 v2))
       (0, ne)
       (zip flags vals)
  |> unzip
  |> (.1)

let exSgmScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (flags: [n]i64)
            (vals: [n]t)
            : [n]t =
  exscan (\(f1, v1) (f2, v2) -> (if f1 == 0 || f2 == 0 then 0 else 1, if f2 != 0 then v2 else op v1 v2))
       (0, ne)
       (zip flags vals)
  |> unzip
  |> (.1)

let primesFlat (n: i64) : []i64 =
  let sq_primes = [2i64, 3i64, 5i64, 7i64]
  let len = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\p -> (len / p) - 1) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 

      let fmm1  = mult_lens
      let inds  = exscan (+) 0 fmm1
      let flag  = scatter (replicate flat_size 0) inds fmm1
      
      -- mkFlagArray inds 0 fmm1 :> [flat_size]i64

      -- Iota nested in map
      let fiot  = exSgmScan (+) 0 flag (replicate flat_size 1)

      -- Map nested in map
      let ftwom = map (+2) fiot

      -- Replicate nested in map
      let vals  = scatter (replicate flat_size 0) inds sq_primes
      let frp   = sgmScan (+) 0 flag vals

      -- Map nested in map
      let not_primes = map (\(j, p) -> j * p) (zip ftwom frp)

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array  = replicate flat_size 0i8
       let mostly_ones = map (i8.bool <-< (> 1)) (iota (len + 1))
       let prime_flags = scatter mostly_ones not_primes zero_array
       let sq_primes   = filter (\i -> #[unsafe] prime_flags[i] != 0) (iota (len + 1))
       in (sq_primes, len)
  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n: i64) : []i64 = primesFlat n
