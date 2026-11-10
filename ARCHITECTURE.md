# VCSR-MC architecture and review

Reviewed at commit `66bac60`. This is analysis of the source, the rocSPARSE
source and compiler output. The repository contains no measured results, and
none are reported here.

## Data path

```text
Matrix Market A                        dense FP32 B [K, O]
       |                                       |
sorted CSR [M, K]                              |
       | host format conversion               |
segments of S original columns                 |
       |                                       |
active rows sorted by segment nnz              |
       |                                       |
16-row groups, depth-major slots               |
       |                                       |
same-segment work items (1..32 groups)         |
       |                                       |
       +--> one block per (item, T_k output columns)
                       |
          load B segment (S x T_k) into padded LDS
          reuse its fragments across the item's groups
                       |
          scatter packed A into a 16 x S LDS tile
          (three rotating A buffers, zeros in gaps)
                       |
            wave64 MFMA, FP32 accumulators
                       |
            atomicAdd segment partials into C
            (requires C = 0 before every call)
```

A packed depth slot holds each row's d-th stored entry, so its 16 lanes can
have different original column offsets. The kernel rebuilds those offsets in
LDS before the MFMA. B and C are row-major.

## Default configuration: S=32, T_k=128, O=128

`T_k` is the `--tileK` flag: the **output-column tile width**, not reduction K.
Compiler figures are from ROCm 7.2.4 (`rocm/dev-ubuntu-24.04:7.2.4`,
hipSPARSE/rocSPARSE 4.2.0, `-Rpass-analysis=kernel-resource-usage`), compile
only.

| | FP16 path (`vcsr_mc`) | FP32 path (`vcsr_mc_fp32`) |
|---|---|---|
| Threads/block | 512 = 8 wavefronts x 64 | same |
| Work per wave per group | 16 rows x 16 output columns | same |
| MFMA per wave per group | 2 x `v_mfma_f32_16x16x16_f16` | 8 x `v_mfma_f32_16x16x4_f32` |
| Accumulation and output | FP32 | FP32 |
| LDS/block | 11,904 B | 24,960 B |
| VGPRs / spills | 26 / none | 30 / none |
| Compiler occupancy | 8 waves/SIMD (wave-slot bound) | 4 waves/SIMD (LDS bound) |
| Output atomic | native `global_atomic_add_f32` | same |
| Stored A values, B input | FP32, converted to FP16 in LDS | FP32 |

The FP32 atomics lower to the hardware instruction without
`-munsafe-fp-atomics`, so there is no CAS loop. The B-tile `float4` fast path
merges with the scalar tail path into `global_load_dwordx3` plus
`global_load_dword` rather than one `dwordx4`. The FP16 kernel gathers B
fragments with eight `ds_read_u16` and scatters A with `ds_write_b16`.

MI300A is CDNA3/gfx942 with 228 CUs and 64 KiB LDS per CU. MI300X shares the
gfx942 target but has different resources, so its results do not transfer.

## Comparison with the rocSPARSE baseline

Read from `ROCm/rocm-libraries` tag `rocm-7.2.4` (`dabb6df2`). Kernel selection
can differ in other ROCm versions.

**Reached path.** `hipsparseSpMM(CSR, HIPSPARSE_SPMM_ALG_DEFAULT)` is a thin
wrapper over `rocsparse_spmm(..., compute)`. Default maps to
`rocsparse_csrmm_alg_default`, which runs `csrmm_template_row_split`
(`rocsparse_spmm.cpp:151-178`, `rocsparse_csrmm_impl.cpp:141-175`). hipSPARSE
maps `CSR_ALG1` to row-split and `CSR_ALG2/3` to `rocsparse_spmm_alg_csr`, which
also resolves to row-split (`hipsparse/.../amd_detail/utility.h:508-536`).
rocSPARSE's `nnz_split` and `merge_path` are reachable only through native
`rocsparse_spmm`. All three have FP16-input/FP32-compute instances.

For row-major B without transpose and O=128, row-split launches one kernel,
`csrmmnt_row_split_subwfsize_x_loop_columns_kernel<256,16,8,8>`
(`rocsparse_csrmm_template_row_split.cpp:304-316`):

| | rocSPARSE row-split | VCSR-MC |
|---|---|---|
| Work unit | one CSR row per 16 lanes, 16 rows per 256-thread block | 16-row group x S columns, 4·T_k threads |
| Arithmetic | scalar FMA in FP32 (FP16 inputs widened) | MFMA on zero-filled 16 x S tiles |
| A reads | col/val reread once per 64-column pass: 16 B/nnz at O=128 | 8 B per padded slot, once per output tile |
| B reads | one row of T_k columns per nonzero through L2/Infinity Cache: 4·T_k B FP32, 2·T_k B FP16 | one S x T_k FP32 tile per work item into LDS |
| C writes | each row once, plain stores; beta=0 writes without reading C | O atomics per (row, segment) after a full clear |
| Setup | none | host format construction and upload |

Consequences:

- AMD's MI300A data sheet lists 122.6 TFLOPS for both FP32 vector and FP32
  matrix, and 980.6 TFLOPS dense FP16 matrix. `vcsr_mc_fp32` has no
  peak-arithmetic advantage over row-split's vector FMAs, so an FP32 speedup
  must come from data movement, mainly B reuse through LDS.
- At O=128 the operation is normally bandwidth-bound: row-split does 256 FLOP
  per 512 B of FP32 B. VCSR reads less B than row-split only when a work item
  holds more than **S nonzeros** (2S against `rocsparse_fp16`), because its
  S x T_k tile costs as much as S row-split nonzeros. Matrices with fewer than
  8,192 groups get one group per item (`vcsr_mc.cu:111`), so at S=32 a group
  needs about 2 nonzeros per row per segment. Expect gains on banded, FEM and
  blocky matrices and losses on scale-free graphs. Cache reuse lowers
  row-split's real traffic, so this is a traffic bound, not a time prediction.
- Row-split never clears C or issues atomics. `nnz_split` and `merge_path`
  scale C first (`scale_2d_array`) and then accumulate atomically, the same
  output structure VCSR uses.
- `main.cpp:242` clears C outside the timed events, so VCSR times exclude a
  clear its atomics require. rocSPARSE with beta=0 needs none.
- The matched FP16 comparison is conservative for VCSR in bytes: rocSPARSE's
  mixed instance reads FP16 A and B, while VCSR reads FP32 and converts in LDS.

A rocSPARSE comparison should call native `rocsparse_spmm` with row-split,
nnz-split and merge-path, including their preprocessing, and report each plus
the best per matrix.

## Settings: segment width S and output tile T_k

Both are template parameters chosen at run time based on vcsr-tc

| Setting | Flag | Supported values | Constraint |
|---|---|---|---|
| S | `--segw` | 16, 32, 48, 64 | Multiple of 16: FP16 MFMA consumes 16 columns per instruction (S/16 per group); FP32 consumes 4 (S/4) |
| T_k | `--tileK` | 16, 32, 64, 128 | Multiple of 16: each wavefront owns 16 output columns, so threads/block = 4·T_k |

The binary defaults are S=16, T_k=64 (`main.cpp:70`). `run.sh` uses S=32,
T_k=128.

**S** sets how many segments a row spans. A larger S:

- reduces the (row, segment) pairs, hence the atomics into C (O per pair) and
  the partial sums per row;
- puts more nonzeros in each group, so each B tile serves more of them;
- costs dense MFMA work of 16 x S x T_k per group regardless of nnz (useful
  fraction is about group_nnz / (16·S)), plus LDS and VGPRs linear in S.

**T_k** sets the block shape. The grid has ceil(O / T_k) output tiles, and each
tile rereads and rescatters its work item's A, so A traffic scales with
O / T_k. The atomic count does not depend on T_k. The three A buffers do not
shrink with T_k, so smaller tiles lower occupancy.

Compiler occupancy (waves/SIMD) and LDS bytes per block, every instance in
`run_vcsr_mc_spmm_gpu`. VGPRs depend only on S: 24/26/28/30 (FP16) and
26/30/34/38 (FP32), with no spills anywhere.

FP16 `vcsr_mc`:

| S \ T_k | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 16 | 7, 2,560 | 8, 3,072 | 8, 4,096 | 8, 6,144 |
| 32 | 4, 4,736 | 6, 5,760 | 8, 7,808 | 8, 11,904 |
| 48 | 3, 6,912 | 4, 8,448 | 5, 11,520 | 6, 17,664 |
| 64 | 2, 9,088 | 3, 11,136 | 4, 15,232 | 4, 23,424 |

FP32 `vcsr_mc_fp32`:

| S \ T_k | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 16 | 3, 5,504 | 5, 6,528 | 7, 8,576 | 8, 12,672 |
| 32 | 2, 10,624 | 3, 12,672 | 3, 16,768 | 4, 24,960 |
| 48 | 1, 15,744 | 2, 18,816 | 2, 24,960 | 2, 37,248 |
| 64 | 1, 20,864 | 1, 24,960 | 1, 33,152 | 2, 49,536 |

Keep T_k = 128 (or O when smaller) unless measurements disagree, and sweep S
per matrix. FP32 at S >= 48 runs at 2 waves/SIMD or fewer.

LDS per block is `3·16·(S+4)·2 + S·(T_k+4)·2` bytes for FP16 and
`3·16·(S+2)·4 + S·(T_k+16)·4` for FP32. Larger multiples of 16 are
possible but not instantiated. At S=128, FP16 fits with T_k=128 (46,464 B), and
FP32 needs T_k <= 32 (49,536 B). Adding a value means updating the checks in
`csr_to_vcsr_mc` and `main.cpp` and the instance list in
`run_vcsr_mc_spmm_gpu`, which falls back to S=16 for unlisted values
(`vcsr_mc.cu:378`).

## Improvements

1. **Store A as FP16 values with 1-byte column offsets.** Every padded slot now
   stores an int32 offset and an FP32 value, 8 B in both modes, and the FP16
   kernel converts in LDS. FP16 plus `uint8` is 3 B per slot and covers S <= 256.
   A is reread once per output tile, so the saving grows at small T_k. Keep a
   sentinel such as `0xFF` for padding: writing padding as offset 0 and value 0
   would race with a real entry at offset 0 in the same scatter pass. Compare
   this against `rocsparse_fp16`, which already reads FP16 A and B. Keep FP32
   values for `vcsr_mc_fp32`.
2. **Dispatch on local density.** The host format builder already knows each
   group's nnz, padding and work-item size. Compute the useful MFMA fraction,
   nonzeros per work item against the S break-even above, and segments per
   row. Route matrices, or individual sparse segments, that fall below a
   threshold to rocSPARSE or a CSR kernel. A per-segment hybrid must accumulate
   into the same C (atomics or a beta=1 pass). Calibrate thresholds against
   measured per-matrix MI300A speedups; emitting these statistics in the CSV is
   the first step.

## Proposal: CLI flags and documentation

| Command | Today | Proposed |
|---|---|---|
| `spmm_vcsr` | `--segw` and `--tileK` exist. `--help` lists defaults S=16, T_k=64 but not valid S values. An invalid S is skipped per kernel with a stderr note and no CSV row (`main.cpp:352-355`). | Validate `--segw` against {16, 32, 48, 64} before reading the matrix and exit non-zero. List valid values and defaults in `--help`. Add `--tile-cols` as a clearer alias for `--tileK`. Consider defaults matching `run.sh`. |
| `spmm_vcsr` | No format statistics. | `--stats`, or extra CSV columns: groups, work items, chunk, padded slots, useful fraction, segments per row. These feed improvement 2. |
| `run.sh` | Settings come from environment variables `SEGW`, `TILEK`, `OS`, `GPU`, `ALGOS`, `REPS`. `TAG` defaults to `MI300A_S32T128` whatever S and T_k are (`run.sh:8`), so the result file name can misstate its configuration. | Add `--segw`, `--tileK`, `--O`, `--gpu`, `--algos`, `--reps` and `--help`, keeping the variables as defaults. Accept lists for sweeps. Derive `TAG` from the actual S and T_k. |
| rocSPARSE baseline | `run_rocsparse_spmm` takes `alg_id`, but no flag reaches it, and every hipSPARSE CSR value runs row-split. | `--rocsparse-alg row_split\|nnz_split\|merge_path`, implemented with native `rocsparse_spmm`. |
| `README.md` | One example command. | A settings table with flag, meaning, valid values, default and the occupancy tradeoffs above. |

Proposed usage:

```bash
./run.sh --segw "32 64" --tileK 128 --O 128
build/spmm_vcsr --mtx matrices200/mtx/enron.mtx --algo vcsr_mc,rocsparse_fp16 \
    --segw 64 --tile-cols 128 --stats --stat median
```

Sources: [AMD Instinct MI300A data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf),
[AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html),
[rocm-libraries `rocm-7.2.4`](https://github.com/ROCm/rocm-libraries/tree/rocm-7.2.4/projects/rocsparse/library/src/level3),
`src/vcsr_mc.cu`, `src/kernels.cu`, `src/main.cpp`, `run.sh`.
