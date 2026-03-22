#include "amdgpu.h"
#include "encode.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/*
 * AMDGPU emitter: phi elimination, register allocation, assembly printer,
 * and ELF code object writer.
 * Targets CDNA 2 (gfx90a, Wave64), RDNA 2/3/4 (gfx1030/1100/1200, Wave32).
 * Dependencies: libc, optimism, tea.
 */

/* ---- Phi Elimination ---- */

/*
 * Proper phi elimination: insert copies at predecessor block ends.
 * For each PSEUDO_PHI with operands [(pred, src), ...], insert
 * PSEUDO_COPY dst, srcN at the end of each predecessor (before its
 * terminator). Then NOP the PHI. Cycles are mercifully rare from our
 * frontend, so we don't break them — yet.
 */

#define PHI_MAX_COPIES 4096

typedef struct {
    uint32_t   pred_mb;
    moperand_t dst;
    moperand_t src;
} phi_copy_t;

static phi_copy_t phi_copies[PHI_MAX_COPIES];

/* Is this instruction a block terminator? */
static int is_terminator(uint16_t op)
{
    return op == AMD_S_BRANCH || op == AMD_S_CBRANCH_SCC0 ||
           op == AMD_S_CBRANCH_SCC1 || op == AMD_S_CBRANCH_EXECZ ||
           op == AMD_S_CBRANCH_EXECNZ || op == AMD_S_ENDPGM ||
           op == AMD_S_SETPC_B64;
}

void amdgpu_phi_elim(amd_module_t *A)
{
    uint32_t nc = 0;

    /* Phase 1: collect copies from PHIs, NOP the PHIs */
    for (uint32_t i = 0; i < A->num_minsts; i++) {
        minst_t *mi = &A->minsts[i];
        if (mi->op != AMD_PSEUDO_PHI) continue;

        moperand_t dst = mi->operands[0];
        for (uint8_t p = 0; p + 1 < mi->num_uses && nc < PHI_MAX_COPIES; p += 2) {
            uint32_t off = mi->num_defs + p;
            if (off + 1 >= MINST_MAX_OPS) break;
            if (mi->operands[off].kind != MOP_LABEL) continue;

            phi_copies[nc].pred_mb = (uint32_t)mi->operands[off].imm;
            phi_copies[nc].dst = dst;
            phi_copies[nc].src = mi->operands[off + 1];
            nc++;
        }
        mi->op = AMD_S_NOP;
        mi->num_defs = 0;
        mi->num_uses = 0;
    }

    if (nc == 0) return;

    /* Phase 2: count copies per predecessor block */
    static uint32_t cpb[AMD_MAX_MBLOCKS]; /* copies per block */
    memset(cpb, 0, A->num_mblocks * sizeof(uint32_t));
    for (uint32_t i = 0; i < nc; i++) {
        if (phi_copies[i].pred_mb < A->num_mblocks)
            cpb[phi_copies[i].pred_mb]++;
    }

    /* Phase 3: insert copies before terminators, processing blocks in
       reverse order so shifts don't affect already-processed blocks. */
    for (uint32_t mb = A->num_mblocks; mb > 0; mb--) {
        uint32_t b = mb - 1;
        uint32_t copies_here = cpb[b];
        if (copies_here == 0) continue;
        if (A->num_minsts + copies_here > AMD_MAX_MINSTS) continue;

        mblock_t *B = &A->mblocks[b];

        /* Find insertion point: before trailing terminators */
        uint32_t insert_rel = B->num_insts;
        for (uint32_t ii = B->num_insts; ii > 0; ii--) {
            if (is_terminator(A->minsts[B->first_inst + ii - 1].op))
                insert_rel = ii - 1;
            else
                break;
        }
        uint32_t insert_abs = B->first_inst + insert_rel;

        /* Shift tail of instruction array to make room */
        uint32_t tail_len = A->num_minsts - insert_abs;
        memmove(&A->minsts[insert_abs + copies_here],
                &A->minsts[insert_abs],
                tail_len * sizeof(minst_t));

        /* Insert copies */
        uint32_t ci = 0;
        for (uint32_t i = 0; i < nc && ci < copies_here; i++) {
            if (phi_copies[i].pred_mb != b) continue;
            minst_t *copy = &A->minsts[insert_abs + ci];
            memset(copy, 0, sizeof(minst_t));
            copy->op = AMD_PSEUDO_COPY;
            copy->num_defs = 1;
            copy->num_uses = 1;
            copy->operands[0] = phi_copies[i].dst;
            copy->operands[1] = phi_copies[i].src;
            ci++;
        }

        A->num_minsts += copies_here;
        B->num_insts += copies_here;

        /* Update first_inst for all subsequent blocks */
        for (uint32_t later = b + 1; later < A->num_mblocks; later++)
            A->mblocks[later].first_inst += copies_here;
    }
}

/* ---- Register Allocation ---- */

/* Live interval for a virtual register */
typedef struct {
    uint32_t vreg;
    uint32_t start;    /* first def */
    uint32_t end;      /* last use */
    uint16_t phys;     /* allocated physical reg */
    uint8_t  file;     /* 0=SGPR, 1=VGPR */
    uint8_t  spilled;
} live_interval_t;

/* Static storage for regalloc (~4 MB) */
static struct {
    live_interval_t intervals[AMD_MAX_VREGS];
    uint32_t        num_intervals;

    /* Sort index */
    uint32_t        sorted[AMD_MAX_VREGS];

    /* Free register pools */
    uint8_t         sgpr_free[AMD_MAX_SGPRS];
    uint8_t         vgpr_free[AMD_MAX_VGPRS];
    uint32_t        num_sgpr_free;
    uint32_t        num_vgpr_free;

    /* Active intervals sorted by end point */
    uint32_t        active[AMD_MAX_VREGS];
    uint32_t        num_active;

    /* Track max used */
    uint16_t        max_sgpr;
    uint16_t        max_vgpr;
} RA;

static int interval_cmp_start(const void *a, const void *b)
{
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (RA.intervals[ia].start != RA.intervals[ib].start)
        return (RA.intervals[ia].start < RA.intervals[ib].start) ? -1 : 1;
    return 0;
}

/* Get the vreg referenced by an operand, or 0xFFFF if not a vreg */
static uint16_t operand_vreg(const moperand_t *op)
{
    if (op->kind == MOP_VREG_S || op->kind == MOP_VREG_V)
        return op->reg_num;
    return 0xFFFF;
}

static void compute_live_intervals(const amd_module_t *A, const mfunc_t *F)
{
    RA.num_intervals = 0;

    /* Initialize: one interval per vreg, with start=MAX, end=0 */
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        RA.intervals[v].vreg = v;
        RA.intervals[v].start = 0xFFFFFFFF;
        RA.intervals[v].end = 0;
        RA.intervals[v].phys = 0xFFFF;
        RA.intervals[v].file = A->reg_file[v];
        RA.intervals[v].spilled = 0;
    }

    /* Walk all instructions in the function */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            const minst_t *mi = &A->minsts[mi_idx];

            /* Defs */
            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t vr = operand_vreg(&mi->operands[d]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }

            /* Uses */
            for (uint8_t u = mi->num_defs; u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t vr = operand_vreg(&mi->operands[u]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }
        }
    }

    /* Collect valid intervals */
    RA.num_intervals = 0;
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        if (RA.intervals[v].start != 0xFFFFFFFF) {
            RA.sorted[RA.num_intervals++] = v;
        }
    }

    /* Sort by start point */
    qsort(RA.sorted, RA.num_intervals, sizeof(uint32_t), interval_cmp_start);
}

static void expire_old(uint32_t point)
{
    /* Remove intervals that have ended before this point */
    uint32_t j = 0;
    for (uint32_t i = 0; i < RA.num_active; i++) {
        uint32_t v = RA.active[i];
        if (RA.intervals[v].end >= point) {
            RA.active[j++] = v;
        } else {
            /* Free the register */
            uint16_t phys = RA.intervals[v].phys;
            if (RA.intervals[v].file == 0 && phys < AMD_MAX_SGPRS) {
                RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)phys;
            } else if (phys < AMD_MAX_VGPRS) {
                RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)phys;
            }
        }
    }
    RA.num_active = j;
}

static void rewrite_operands(amd_module_t *A, const mfunc_t *F)
{
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = 0; k < total; k++) {
                moperand_t *op = &mi->operands[k];
                if (op->kind == MOP_VREG_S) {
                    op->kind = MOP_SGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                } else if (op->kind == MOP_VREG_V) {
                    op->kind = MOP_VGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                }
            }

            if (mi->op == AMD_PSEUDO_COPY) {
                if (mi->operands[0].kind == MOP_VGPR)
                    mi->op = AMD_V_MOV_B32;
                else
                    mi->op = AMD_S_MOV_B32;
            }
        }
    }
}

/* Dead copy elimination: kill MOVs where src == dst. */
static void dead_copy_elim(amd_module_t *A, const mfunc_t *F)
{
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            if ((mi->op == AMD_V_MOV_B32 || mi->op == AMD_S_MOV_B32) &&
                mi->num_defs == 1 && mi->num_uses == 1 &&
                mi->operands[0].kind == mi->operands[1].kind &&
                mi->operands[0].reg_num == mi->operands[1].reg_num) {
                mi->op = AMD_PSEUDO_DEF;
                mi->num_defs = 0;
                mi->num_uses = 0;
                continue;
            }

            if ((mi->op == AMD_V_MOV_B32 || mi->op == AMD_S_MOV_B32 ||
                 mi->op == AMD_PSEUDO_COPY) &&
                mi->num_defs == 1 && ii + 1 < MB->num_insts) {
                const minst_t *next = &A->minsts[MB->first_inst + ii + 1];
                uint16_t dst_reg = mi->operands[0].reg_num;
                uint8_t  dst_kind = mi->operands[0].kind;

                int next_uses = 0;
                for (uint8_t u = next->num_defs;
                     u < next->num_defs + next->num_uses && u < MINST_MAX_OPS; u++) {
                    if (next->operands[u].kind == dst_kind &&
                        next->operands[u].reg_num == dst_reg)
                        next_uses = 1;
                }

                int next_defs = 0;
                for (uint8_t d = 0; d < next->num_defs && d < MINST_MAX_OPS; d++) {
                    if (next->operands[d].kind == dst_kind &&
                        next->operands[d].reg_num == dst_reg)
                        next_defs = 1;
                }

                if (next_defs && !next_uses) {
                    mi->op = AMD_PSEUDO_DEF;
                    mi->num_defs = 0;
                    mi->num_uses = 0;
                }
            }
        }
    }
}

static void finalize_reg_counts(const amd_module_t *A, mfunc_t *F)
{
    if (F->num_sgprs == 0) F->num_sgprs = 1;
    if (F->num_vgprs == 0) F->num_vgprs = 1;

    if (F->launch_bounds_max > 0 && F->launch_bounds_max < 1024) {
        int w64 = (A->target <= AMD_TARGET_GFX950);
        uint32_t wsz = w64 ? 64u : 32u;
        uint32_t desired_waves = (F->launch_bounds_max + wsz - 1) / wsz;
        if (desired_waves > 0) {
            uint32_t gran = w64 ? ~3u : ~7u;
            uint32_t vgpr_cap = (256 / desired_waves) & gran;
            if (vgpr_cap < (w64 ? 4u : 8u)) vgpr_cap = w64 ? 4u : 8u;
            if (F->num_vgprs > vgpr_cap)
                F->num_vgprs = (uint16_t)vgpr_cap;
        }
    }
}

/* ---- Linear Scan (fallback) ---- */

static void regalloc_linear(amd_module_t *A, uint32_t mf_idx)
{
    mfunc_t *F = &A->mfuncs[mf_idx];

    /* Initialize free pools */
    RA.num_sgpr_free = 0;
    RA.num_vgpr_free = 0;
    RA.max_sgpr = 0;
    RA.max_vgpr = 0;
    RA.num_active = 0;

    /* Push high regs first so low regs are popped first (stack order) */
    uint16_t sgpr_start = F->is_kernel ? F->first_alloc_sgpr : 0;
    if (sgpr_start < AMD_KERN_RESERVED_SGPR && F->is_kernel)
        sgpr_start = AMD_KERN_RESERVED_SGPR;
    for (uint16_t r = AMD_MAX_SGPRS; r-- > sgpr_start; ) {
        /* gfx950: s10/s11 crash on SMEM loads — exclude globally */
        if (A->target == AMD_TARGET_GFX950 && (r == 10 || r == 11))
            continue;
        RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)r;
    }
    for (uint16_t r = AMD_MAX_VGPRS; r-- > 0; )
        RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)r;

    compute_live_intervals(A, F);

    /* Linear scan */
    for (uint32_t i = 0; i < RA.num_intervals; i++) {
        uint32_t v = RA.sorted[i];
        live_interval_t *iv = &RA.intervals[v];

        expire_old(iv->start);

        uint16_t phys = 0xFFFF;
        if (iv->file == 0) {
            /* SGPR */
            if (RA.num_sgpr_free > 0) {
                phys = RA.sgpr_free[--RA.num_sgpr_free];
                if (phys >= RA.max_sgpr) RA.max_sgpr = phys + 1;
            }
        } else {
            /* VGPR */
            if (RA.num_vgpr_free > 0) {
                phys = RA.vgpr_free[--RA.num_vgpr_free];
                if (phys >= RA.max_vgpr) RA.max_vgpr = phys + 1;
            }
        }

        if (phys == 0xFFFF) {
            /* Spill: find the active interval with the farthest end */
            uint32_t farthest = 0, farthest_idx = 0;
            for (uint32_t a = 0; a < RA.num_active; a++) {
                uint32_t av = RA.active[a];
                if (RA.intervals[av].file == iv->file &&
                    RA.intervals[av].end > farthest) {
                    farthest = RA.intervals[av].end;
                    farthest_idx = a;
                }
            }
            if (farthest > iv->end && RA.num_active > 0) {
                /* Spill the farthest, give its reg to us */
                uint32_t sv = RA.active[farthest_idx];
                phys = RA.intervals[sv].phys;
                RA.intervals[sv].spilled = 1;
                RA.intervals[sv].phys = 0xFFFF;
                /* Remove from active */
                RA.active[farthest_idx] = RA.active[--RA.num_active];
            } else {
                /* Spill ourselves */
                iv->spilled = 1;
                phys = 0; /* fallback */
            }
        }

        iv->phys = phys;
        A->reg_map[v] = phys;

        /* Add to active */
        if (RA.num_active < AMD_MAX_VREGS)
            RA.active[RA.num_active++] = v;
    }

    /* Record usage for kernel descriptor */
    F->num_sgprs = RA.max_sgpr;
    F->num_vgprs = RA.max_vgpr;
    finalize_reg_counts(A, F);
    rewrite_operands(A, F);
    dead_copy_elim(A, F);
}

/* ---- Graph Coloring Register Allocation ---- */

#define RA_MAX_NODES 8192

typedef struct {
    uint32_t vreg;
    uint32_t degree;
    float    spill_cost;
    uint16_t color;
    uint16_t alias;
    uint8_t  file;
    uint8_t  spilled;
    uint8_t  in_graph;
} ra_node_t;

/* Dynamically allocated per regalloc_graphcolor() call, sized to actual
   node count N.  NOTE: these static globals make the allocator non-reentrant.  Fine for
   single-threaded compilation; would need to be bundled into a context
   struct if we ever parallelize across functions. */
static void      *ra_mem;
static uint32_t  *ra_ifg;
static ra_node_t *ra_nodes;
static uint32_t  *ra_stack;
static uint32_t   ra_num_nodes;
static uint32_t   ra_stride;
static uint32_t   ra_stack_top;

static uint16_t ra_vreg_to_node[AMD_MAX_VREGS];

static void ra_ifg_set(uint32_t i, uint32_t j)
{
    uint64_t bit;
    bit = (uint64_t)i * ra_stride + j;
    ra_ifg[bit / 32] |= 1u << (bit % 32);
    bit = (uint64_t)j * ra_stride + i;
    ra_ifg[bit / 32] |= 1u << (bit % 32);
}

static int ra_ifg_test(uint32_t i, uint32_t j)
{
    uint64_t bit = (uint64_t)i * ra_stride + j;
    return (ra_ifg[bit / 32] >> (bit % 32)) & 1;
}

static uint16_t ra_find(uint16_t n)
{
    while (ra_nodes[n].alias != n) {
        ra_nodes[n].alias = ra_nodes[ra_nodes[n].alias].alias;
        n = ra_nodes[n].alias;
    }
    return n;
}

static void ra_coalesce(uint16_t a, uint16_t b)
{
    ra_nodes[b].alias = a;
    for (uint32_t k = 0; k < ra_num_nodes; k++) {
        if (k == a || k == b) continue;
        if (ra_find((uint16_t)k) != k) continue;
        if (ra_ifg_test(b, k)) {
            if (!ra_ifg_test(a, k)) {
                ra_ifg_set(a, k);
                ra_nodes[a].degree++;
            }
        }
    }
    ra_nodes[b].in_graph = 0;
}

/* Bitvector operations for liveness sets.
   One bit per vreg, stored as uint32_t words. */

static void bv_set(uint32_t *bv, uint32_t bit)
{
    bv[bit / 32] |= 1u << (bit % 32);
}

static void bv_clear(uint32_t *bv, uint32_t bit)
{
    bv[bit / 32] &= ~(1u << (bit % 32));
}

static int bv_test(const uint32_t *bv, uint32_t bit)
{
    return (int)((bv[bit / 32] >> (bit % 32)) & 1u);
}

static int bv_or(uint32_t *dst, const uint32_t *src, uint32_t nwords)
{
    int changed = 0;
    for (uint32_t w = 0; w < nwords; w++) {
        uint32_t old = dst[w];
        dst[w] |= src[w];
        if (dst[w] != old) changed = 1;
    }
    return changed;
}

/* Per-block liveness arrays.  Dynamically allocated, sized to actual
   num_blocks × bv_words.  Accessed as ra_live_in[bi * ra_bv_words + w]. */
#define RA_MAX_BLOCKS 4096

static uint32_t *ra_live_in;    /* [nb][bv_words] flattened */
static uint32_t *ra_live_out;
static uint32_t *ra_blk_def;
static uint32_t *ra_blk_use;
static uint16_t *ra_succs;      /* [nb][2] flattened */
static uint8_t  *ra_nsuccs;     /* [nb] */
static uint32_t  ra_bv_words;   /* bitvector width for current allocation */
static void     *ra_liveness_mem;

#define RA_BV(arr, bi)  ((arr) + (size_t)(bi) * ra_bv_words)
#define RA_SUCCS(bi)    (ra_succs + (size_t)(bi) * 2)

static void ra_build_cfg(const amd_module_t *A, const mfunc_t *F)
{
    uint32_t nb = F->num_blocks;
    memset(ra_nsuccs, 0, nb * sizeof(uint8_t));

    for (uint32_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        int has_unconditional = 0;

        for (uint32_t ii = MB->num_insts; ii > 0; ii--) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii - 1];
            if (!is_terminator(mi->op)) break;

            if (mi->op == AMD_S_ENDPGM || mi->op == AMD_S_SETPC_B64) {
                has_unconditional = 1;
            } else if (mi->op == AMD_S_BRANCH) {
                has_unconditional = 1;
                if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
                    uint32_t target = (uint32_t)mi->operands[0].imm;
                    if (target >= F->first_block &&
                        target < F->first_block + nb &&
                        ra_nsuccs[bi] < 2) {
                        RA_SUCCS(bi)[ra_nsuccs[bi]++] =
                            (uint16_t)(target - F->first_block);
                    }
                }
            } else if (mi->op == AMD_S_CBRANCH_SCC0 ||
                       mi->op == AMD_S_CBRANCH_SCC1 ||
                       mi->op == AMD_S_CBRANCH_EXECZ ||
                       mi->op == AMD_S_CBRANCH_EXECNZ) {
                if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
                    uint32_t target = (uint32_t)mi->operands[0].imm;
                    if (target >= F->first_block &&
                        target < F->first_block + nb &&
                        ra_nsuccs[bi] < 2) {
                        RA_SUCCS(bi)[ra_nsuccs[bi]++] =
                            (uint16_t)(target - F->first_block);
                    }
                }
            }
        }

        /* Fallthrough: if block doesn't end with an unconditional branch
           or endpgm, the next block is a successor */
        if (!has_unconditional && bi + 1 < nb && ra_nsuccs[bi] < 2) {
            RA_SUCCS(bi)[ra_nsuccs[bi]++] = (uint16_t)(bi + 1);
        }
    }
}

/* Compute per-block def/use sets, then solve live_in/live_out via
   backward dataflow iteration:
     live_in[B]  = use[B] ∪ (live_out[B] − def[B])
     live_out[B] = ∪ { live_in[S] : S ∈ successors(B) }
   Iterates until fixpoint. */
static void ra_compute_liveness(const amd_module_t *A, const mfunc_t *F,
                                uint32_t nv)
{
    uint32_t nb = F->num_blocks;
    uint32_t bv_words = ra_bv_words;
    for (uint32_t bi = 0; bi < nb; bi++) {
        memset(RA_BV(ra_blk_def, bi), 0, bv_words * sizeof(uint32_t));
        memset(RA_BV(ra_blk_use, bi), 0, bv_words * sizeof(uint32_t));

        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];

            for (uint8_t u = mi->num_defs;
                 u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t vr = operand_vreg(&mi->operands[u]);
                if (vr != 0xFFFF && vr < nv && !bv_test(RA_BV(ra_blk_def, bi), vr))
                    bv_set(RA_BV(ra_blk_use, bi), vr);
            }
            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t vr = operand_vreg(&mi->operands[d]);
                if (vr != 0xFFFF && vr < nv)
                    bv_set(RA_BV(ra_blk_def, bi), vr);
            }
        }
    }

    for (uint32_t bi = 0; bi < nb; bi++) {
        memset(RA_BV(ra_live_in, bi),  0, bv_words * sizeof(uint32_t));
        memset(RA_BV(ra_live_out, bi), 0, bv_words * sizeof(uint32_t));
    }

    for (int pass = 0; pass < 200; pass++) {
        int changed = 0;

        for (uint32_t bi2 = nb; bi2 > 0; bi2--) {
            uint32_t bi = bi2 - 1;

            for (uint8_t s = 0; s < ra_nsuccs[bi]; s++) {
                uint16_t succ = RA_SUCCS(bi)[s];
                if (succ < nb)
                    changed |= bv_or(RA_BV(ra_live_out, bi), RA_BV(ra_live_in, succ), bv_words);
            }

            for (uint32_t w = 0; w < bv_words; w++) {
                uint32_t new_in = RA_BV(ra_blk_use, bi)[w] |
                                  (RA_BV(ra_live_out, bi)[w] & ~RA_BV(ra_blk_def, bi)[w]);
                if (new_in != RA_BV(ra_live_in, bi)[w]) {
                    RA_BV(ra_live_in, bi)[w] = new_in;
                    changed = 1;
                }
            }
        }

        if (!changed) break;
    }
}

static void ra_build_ifg_from_liveness(const amd_module_t *A,
                                       const mfunc_t *F,
                                       uint32_t nv)
{
    uint32_t bv_words = ra_bv_words;
    uint32_t *live = (uint32_t *)calloc(bv_words, sizeof(uint32_t));
    if (!live) return;

    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];

        memcpy(live, RA_BV(ra_live_out, bi), bv_words * sizeof(uint32_t));

        for (uint32_t ii = MB->num_insts; ii > 0; ii--) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii - 1];

            /* For PSEUDO_COPY D = S: don't add D↔S interference even
               though S is live at D's def.  This is what makes coalescing
               possible --without it, copy-related pairs always interfere
               and can never be merged. */
            int is_copy = (mi->op == AMD_PSEUDO_COPY);
            uint16_t copy_src = 0xFFFF;
            if (is_copy && mi->num_uses >= 1)
                copy_src = operand_vreg(&mi->operands[mi->num_defs]);

            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t dv = operand_vreg(&mi->operands[d]);
                if (dv == 0xFFFF || dv >= nv) continue;
                uint16_t dn = ra_vreg_to_node[dv];
                if (dn == 0xFFFF) continue;

                for (uint32_t w = 0; w < bv_words; w++) {
                    uint32_t bits = live[w];
                    while (bits) {
                        uint32_t lsb = bits & (uint32_t)(-(int32_t)bits);
                        uint32_t bit_idx = w * 32;
                        /* CTZ -- portable across MSVC/MinGW */
                        { uint32_t tmp = lsb; while (!(tmp & 1)) { tmp >>= 1; bit_idx++; } }
                        bits &= bits - 1;

                        if (bit_idx >= nv) continue;
                        uint16_t lv = (uint16_t)bit_idx;
                        if (lv == dv) continue;
                        if (is_copy && lv == copy_src) continue;
                        uint16_t ln = ra_vreg_to_node[lv];
                        if (ln == 0xFFFF) continue;

                        /* Only interfere within same register file */
                        if (ra_nodes[dn].file != ra_nodes[ln].file) continue;

                        if (!ra_ifg_test(dn, ln)) {
                            ra_ifg_set(dn, ln);
                            ra_nodes[dn].degree++;
                            ra_nodes[ln].degree++;
                        }
                    }
                }

                bv_clear(live, dv);
            }

            for (uint8_t u = mi->num_defs;
                 u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t uv = operand_vreg(&mi->operands[u]);
                if (uv != 0xFFFF && uv < nv)
                    bv_set(live, uv);
            }
        }
    }

    free(live);
}

static void regalloc_graphcolor(amd_module_t *A, uint32_t mf_idx)
{
    mfunc_t *F = &A->mfuncs[mf_idx];
    uint32_t max_iters = 4;
    int gc_success = 0;

    if (F->num_blocks > RA_MAX_BLOCKS) {
        /* Too many blocks --fall back to linear scan */
        regalloc_linear(A, mf_idx);
        return;
    }

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        uint32_t nv = A->vreg_count;
        if (nv > RA_MAX_NODES) nv = RA_MAX_NODES;

        /* Allocate ra_ifg + ra_nodes + ra_stack in one calloc, sized
           to nv (upper bound on node count).  calloc zero-inits the
           bitmatrix and node fields for free. */
        {
            size_t ifg_words = ((uint64_t)nv * nv + 31) / 32;
            size_t total = ifg_words * sizeof(uint32_t)
                         + nv * sizeof(ra_node_t)
                         + nv * sizeof(uint32_t);
            ra_mem = calloc(1, total);
            if (!ra_mem) {
                regalloc_linear(A, mf_idx);
                return;
            }
            ra_ifg   = (uint32_t *)ra_mem;
            ra_nodes = (ra_node_t *)((char *)ra_mem + ifg_words * sizeof(uint32_t));
            ra_stack = (uint32_t *)((char *)ra_nodes + nv * sizeof(ra_node_t));
            ra_stride = nv;
        }

        /* --- Build node table from all vregs used in this function --- */
        ra_num_nodes = 0;
        memset(ra_vreg_to_node, 0xFF, nv * sizeof(uint16_t));

        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                const minst_t *mi = &A->minsts[MB->first_inst + ii];
                uint8_t total = mi->num_defs + mi->num_uses;
                if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;
                for (uint8_t k = 0; k < total; k++) {
                    uint16_t vr = operand_vreg(&mi->operands[k]);
                    if (vr != 0xFFFF && vr < nv) {
                        if (ra_vreg_to_node[vr] == 0xFFFF &&
                            ra_num_nodes < RA_MAX_NODES) {
                            uint16_t nidx = (uint16_t)ra_num_nodes++;
                            ra_nodes[nidx].vreg = vr;
                            ra_nodes[nidx].degree = 0;
                            ra_nodes[nidx].spill_cost = 1.0f;
                            ra_nodes[nidx].color = 0xFFFF;
                            ra_nodes[nidx].alias = nidx;
                            ra_nodes[nidx].file = A->reg_file[vr];
                            ra_nodes[nidx].spilled = 0;
                            ra_nodes[nidx].in_graph = 1;
                            ra_vreg_to_node[vr] = nidx;
                        } else if (ra_vreg_to_node[vr] != 0xFFFF) {
                            ra_nodes[ra_vreg_to_node[vr]].spill_cost += 1.0f;
                        }
                    }
                }
            }
        }

        if (ra_num_nodes == 0) { free(ra_mem); ra_mem = NULL; gc_success = 1; break; }

        /* --- Allocate per-block liveness arrays --- */
        {
            uint32_t nb = F->num_blocks;
            ra_bv_words = (nv + 31) / 32;
            size_t bv_sz = (size_t)nb * ra_bv_words * sizeof(uint32_t);
            size_t succ_sz = (size_t)nb * 2 * sizeof(uint16_t);
            size_t nsucc_sz = (size_t)nb * sizeof(uint8_t);
            size_t total = bv_sz * 4 + succ_sz + nsucc_sz;
            ra_liveness_mem = calloc(1, total);
            if (!ra_liveness_mem) {
                free(ra_mem); ra_mem = NULL;
                regalloc_linear(A, mf_idx);
                return;
            }
            char *p = (char *)ra_liveness_mem;
            ra_live_in  = (uint32_t *)p; p += bv_sz;
            ra_live_out = (uint32_t *)p; p += bv_sz;
            ra_blk_def  = (uint32_t *)p; p += bv_sz;
            ra_blk_use  = (uint32_t *)p; p += bv_sz;
            ra_succs    = (uint16_t *)p; p += succ_sz;
            ra_nsuccs   = (uint8_t  *)p;
        }

        /* --- Build CFG and compute per-block liveness --- */
        ra_build_cfg(A, F);
        ra_compute_liveness(A, F, nv);

        /* --- Build interference graph from liveness --- */
        ra_build_ifg_from_liveness(A, F, nv);

        /* --- Compute K (available physical regs per file) --- */
        uint16_t sgpr_start = F->is_kernel ? F->first_alloc_sgpr : 0;
        if (sgpr_start < AMD_KERN_RESERVED_SGPR && F->is_kernel)
            sgpr_start = AMD_KERN_RESERVED_SGPR;
        uint32_t K_sgpr = (AMD_MAX_SGPRS > sgpr_start) ? AMD_MAX_SGPRS - sgpr_start : 0;
        /* gfx950: 2 fewer SGPRs available (s10/s11 excluded) */
        if (A->target == AMD_TARGET_GFX950 && K_sgpr >= 2) K_sgpr -= 2;
        uint32_t K_vgpr = (amdgpu_max_vgprs > 0 && amdgpu_max_vgprs < AMD_MAX_VGPRS)
                           ? (uint32_t)amdgpu_max_vgprs : AMD_MAX_VGPRS;

        /* --- Copy coalescing (conservative, Briggs criterion) ---
           Only merge two non-interfering copy-related nodes if the
           resulting node would have fewer than K neighbors with
           degree >= K.  This guarantees the merge can't turn a
           colorable graph into one that needs spills --important on
           GPU where a spill means scratch memory (hundreds of cycles
           vs 1 cycle for a register MOV).
           */
        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                uint32_t mi_idx = MB->first_inst + ii;
                const minst_t *mi = &A->minsts[mi_idx];

                if (mi->op != AMD_PSEUDO_COPY) continue;
                if (mi->num_defs < 1 || mi->num_uses < 1) continue;

                uint16_t dst_vr = operand_vreg(&mi->operands[0]);
                uint16_t src_vr = operand_vreg(&mi->operands[1]);
                if (dst_vr == 0xFFFF || src_vr == 0xFFFF) continue;

                uint16_t dn = ra_vreg_to_node[dst_vr];
                uint16_t sn = ra_vreg_to_node[src_vr];
                if (dn == 0xFFFF || sn == 0xFFFF) continue;

                dn = ra_find(dn);
                sn = ra_find(sn);
                if (dn == sn) continue;

                if (ra_nodes[dn].file != ra_nodes[sn].file) continue;
                if (ra_ifg_test(dn, sn)) continue;

                /* Briggs test: count neighbors of the merged node that
                   would have degree >= K.  A neighbor of the merged node
                   is anything adjacent to a OR b (de-duped). */
                uint32_t K = (ra_nodes[dn].file == 0) ? K_sgpr : K_vgpr;
                uint32_t high_deg = 0;
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (ra_find((uint16_t)m) != m) continue;
                    if (m == (uint32_t)dn || m == (uint32_t)sn) continue;
                    if (!ra_nodes[m].in_graph) continue;
                    if (ra_nodes[m].file != ra_nodes[dn].file) continue;
                    if (!ra_ifg_test(dn, m) && !ra_ifg_test(sn, m)) continue;
                    uint32_t mdeg = ra_nodes[m].degree;
                    if (ra_ifg_test(dn, m) && ra_ifg_test(sn, m) && mdeg > 0)
                        mdeg--;
                    if (mdeg >= K) high_deg++;
                }

                if (high_deg < K)
                    ra_coalesce(dn, sn);
            }
        }

        /* --- Recompute degrees after coalescing (only for representatives) --- */
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            if (ra_find((uint16_t)n) != n) { ra_nodes[n].in_graph = 0; continue; }
            if (!ra_nodes[n].in_graph) continue;
            uint32_t deg = 0;
            for (uint32_t m = 0; m < ra_num_nodes; m++) {
                if (m == n) continue;
                if (ra_find((uint16_t)m) != m) continue;
                if (!ra_nodes[m].in_graph) continue;
                if (ra_ifg_test(n, m) && ra_nodes[m].file == ra_nodes[n].file)
                    deg++;
            }
            ra_nodes[n].degree = deg;
        }

        /* --- Simplify: repeatedly remove nodes with degree < K --- */
        ra_stack_top = 0;
        uint32_t nodes_left = 0;
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            if (ra_nodes[n].in_graph) nodes_left++;
        }

        while (nodes_left > 0) {
            int found = 0;
            for (uint32_t n = 0; n < ra_num_nodes; n++) {
                if (!ra_nodes[n].in_graph) continue;
                uint32_t K = (ra_nodes[n].file == 0) ? K_sgpr : K_vgpr;
                if (ra_nodes[n].degree < K) {
                    ra_nodes[n].in_graph = 0;
                    ra_stack[ra_stack_top++] = n;
                    nodes_left--;
                    for (uint32_t m = 0; m < ra_num_nodes; m++) {
                        if (!ra_nodes[m].in_graph) continue;
                        if (ra_ifg_test(n, m) &&
                            ra_nodes[m].file == ra_nodes[n].file &&
                            ra_nodes[m].degree > 0)
                            ra_nodes[m].degree--;
                    }
                    found = 1;
                    break;
                }
            }

            if (!found) {
                /* No simplify candidate --pick lowest spill_cost/degree
                   as potential spill (optimistic: it might still color).
                   Nodes with many uses are expensive to spill; nodes with
                   high degree free the most neighbors when removed. */
                uint32_t best = 0;
                float best_ratio = 1e30f;
                int have_best = 0;
                for (uint32_t n = 0; n < ra_num_nodes; n++) {
                    if (!ra_nodes[n].in_graph) continue;
                    float ratio = ra_nodes[n].spill_cost
                                / (float)(ra_nodes[n].degree + 1);
                    if (!have_best || ratio < best_ratio) {
                        best = n;
                        best_ratio = ratio;
                        have_best = 1;
                    }
                }
                if (!have_best) break;
                ra_nodes[best].in_graph = 0;
                ra_stack[ra_stack_top++] = best;
                nodes_left--;
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_nodes[m].in_graph) continue;
                    if (ra_ifg_test(best, m) &&
                        ra_nodes[m].file == ra_nodes[best].file &&
                        ra_nodes[m].degree > 0)
                        ra_nodes[m].degree--;
                }
            }
        }

        /* --- Select: pop stack, assign colors --- */
        uint16_t max_sgpr = 0, max_vgpr = 0;
        int any_spilled = 0;

        /* Bitvectors for used colors among neighbors */
        static uint32_t used_sgpr[(AMD_MAX_SGPRS + 31) / 32];
        static uint32_t used_vgpr[(AMD_MAX_VGPRS + 31) / 32];

        while (ra_stack_top > 0) {
            uint32_t n = ra_stack[--ra_stack_top];
            uint16_t rep = ra_find((uint16_t)n);
            if (rep != n) {
                /* Coalesced away --will get color from representative */
                continue;
            }

            /* Collect colors used by already-colored neighbors.
               Check interference against original node indices (what the
               matrix was built with), then resolve to the representative
               to get the actual assigned color.  An original edge n-m
               means n interferes with whatever m got coalesced into. */
            if (ra_nodes[n].file == 0) {
                memset(used_sgpr, 0, sizeof(used_sgpr));
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_ifg_test(n, m)) continue;
                    uint16_t mr = ra_find((uint16_t)m);
                    if (mr == (uint16_t)n) continue; /* coalesced into us */
                    if (ra_nodes[mr].color != 0xFFFF && ra_nodes[mr].file == 0) {
                        uint16_t c = ra_nodes[mr].color;
                        used_sgpr[c / 32] |= 1u << (c % 32);
                    }
                }
                uint16_t picked = 0xFFFF;
                for (uint16_t r = sgpr_start; r < AMD_MAX_SGPRS; r++) {
                    /* gfx950: s10/s11 crash on SMEM loads — skip */
                    if (A->target == AMD_TARGET_GFX950 && (r == 10 || r == 11))
                        continue;
                    if (!(used_sgpr[r / 32] & (1u << (r % 32)))) {
                        picked = r;
                        break;
                    }
                }
                if (picked != 0xFFFF) {
                    ra_nodes[n].color = picked;
                    if (picked + 1 > max_sgpr) max_sgpr = picked + 1;
                } else {
                    ra_nodes[n].spilled = 1;
                    any_spilled = 1;
                }
            } else {
                memset(used_vgpr, 0, sizeof(used_vgpr));
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_ifg_test(n, m)) continue;
                    uint16_t mr = ra_find((uint16_t)m);
                    if (mr == (uint16_t)n) continue;
                    if (ra_nodes[mr].color != 0xFFFF && ra_nodes[mr].file == 1) {
                        uint16_t c = ra_nodes[mr].color;
                        used_vgpr[c / 32] |= 1u << (c % 32);
                    }
                }
                uint16_t picked = 0xFFFF;
                for (uint16_t r = 0; r < AMD_MAX_VGPRS; r++) {
                    if (!(used_vgpr[r / 32] & (1u << (r % 32)))) {
                        picked = r;
                        break;
                    }
                }
                if (picked != 0xFFFF) {
                    ra_nodes[n].color = picked;
                    if (picked + 1 > max_vgpr) max_vgpr = picked + 1;
                } else {
                    ra_nodes[n].spilled = 1;
                    any_spilled = 1;
                }
            }
        }

        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            uint16_t rep = ra_find((uint16_t)n);
            if (rep != n) {
                ra_nodes[n].color = ra_nodes[rep].color;
                ra_nodes[n].spilled = ra_nodes[rep].spilled;
            }
        }

        /* --- Spill code insertion ---
           For each block, expand instructions into a full-function output
           buffer, inserting scratch_load before uses and scratch_store after
           defs of spilled vregs.  Then shift the tail once and copy all
           expanded blocks back in a single pass. */
        if (any_spilled) {
            static uint32_t spill_vreg[RA_MAX_NODES];
            static uint8_t  spill_file[RA_MAX_NODES];
            static uint32_t spill_off[RA_MAX_NODES];
            uint32_t num_spills = 0;

            for (uint32_t n = 0; n < ra_num_nodes; n++) {
                if (!ra_nodes[n].spilled) continue;
                if (num_spills >= RA_MAX_NODES) break;
                spill_vreg[num_spills] = ra_nodes[n].vreg;
                spill_file[num_spills] = ra_nodes[n].file;
                spill_off[num_spills] = F->scratch_bytes;
                F->scratch_bytes += 4;
                num_spills++;
            }

            /* Full-function output buffer --sized to hold all instructions
               across all blocks with room for spill loads/stores.
               Single-pass: expand all blocks into this buffer, then
               bulk-copy back once at the end. */
            uint32_t total_insts = 0;
            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++)
                total_insts += A->mblocks[F->first_block + bi2].num_insts;
            /* worst case: each inst gets a load+store per spill, plus
               SGPR spills add an extra intermediary op each */
            size_t out_cap = (size_t)total_insts * (1 + 3 * num_spills) + 16;
            if (out_cap > AMD_MAX_MINSTS) out_cap = AMD_MAX_MINSTS;
            minst_t *ra_output = (minst_t *)malloc(out_cap * sizeof(minst_t));
            if (!ra_output) break;

            uint32_t *blk_out_start = (uint32_t *)malloc(F->num_blocks * sizeof(uint32_t));
            uint32_t *blk_out_count = (uint32_t *)malloc(F->num_blocks * sizeof(uint32_t));
            if (!blk_out_start || !blk_out_count) {
                free(ra_output); free(blk_out_start); free(blk_out_count);
                break;
            }
            uint32_t out_count = 0;

            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++) {
                mblock_t *MB = &A->mblocks[F->first_block + bi2];
                blk_out_start[bi2] = out_count;

                for (uint32_t ii2 = 0; ii2 < MB->num_insts; ii2++) {
                    minst_t orig = A->minsts[MB->first_inst + ii2];

                    for (uint32_t si = 0; si < num_spills; si++) {
                        uint32_t sv = spill_vreg[si];
                        int has_use = 0;
                        for (uint8_t u = orig.num_defs;
                             u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                            if (operand_vreg(&orig.operands[u]) == sv) has_use = 1;
                        }
                        if (has_use) {
                            uint32_t new_vr = A->vreg_count;
                            if (new_vr < AMD_MAX_VREGS) {
                                A->vreg_count++;

                                if (spill_file[si] == 0) {
                                    /* SGPR spill reload: scratch is vector memory,
                                       so load into a temp VGPR then readfirstlane
                                       into the SGPR.
                                       scratch_load_dword vTmp, off, spill_off
                                       v_readfirstlane_b32 sNew, vTmp */
                                    uint32_t vtmp = A->vreg_count;
                                    if (vtmp >= AMD_MAX_VREGS) goto skip_reload;
                                    A->reg_file[vtmp] = 1; /* VGPR */
                                    A->vreg_count++;

                                    A->reg_file[new_vr] = 0; /* SGPR */

                                    for (uint8_t u = orig.num_defs;
                                         u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                                        if (operand_vreg(&orig.operands[u]) == sv)
                                            orig.operands[u].reg_num = (uint16_t)new_vr;
                                    }

                                    /* scratch_load_dword vTmp */
                                    if (out_count < out_cap) {
                                        minst_t *ld = &ra_output[out_count++];
                                        memset(ld, 0, sizeof(minst_t));
                                        ld->op = AMD_SCRATCH_LOAD_DWORD;
                                        ld->num_defs = 1;
                                        ld->num_uses = 2;
                                        ld->operands[0].kind = MOP_VREG_V;
                                        ld->operands[0].reg_num = (uint16_t)vtmp;
                                        ld->operands[1].kind = MOP_IMM;
                                        ld->operands[1].imm = 0;
                                        ld->operands[2].kind = MOP_IMM;
                                        ld->operands[2].imm = (int32_t)spill_off[si];
                                    }
                                    /* v_readfirstlane_b32 sNew, vTmp */
                                    if (out_count < out_cap) {
                                        minst_t *rfl = &ra_output[out_count++];
                                        memset(rfl, 0, sizeof(minst_t));
                                        rfl->op = AMD_V_READFIRSTLANE_B32;
                                        rfl->num_defs = 1;
                                        rfl->num_uses = 1;
                                        rfl->operands[0].kind = MOP_VREG_S;
                                        rfl->operands[0].reg_num = (uint16_t)new_vr;
                                        rfl->operands[1].kind = MOP_VREG_V;
                                        rfl->operands[1].reg_num = (uint16_t)vtmp;
                                    }
                                } else {
                                    A->reg_file[new_vr] = 1; /* VGPR */

                                    for (uint8_t u = orig.num_defs;
                                         u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                                        if (operand_vreg(&orig.operands[u]) == sv)
                                            orig.operands[u].reg_num = (uint16_t)new_vr;
                                    }

                                    if (out_count < out_cap) {
                                        minst_t *ld = &ra_output[out_count++];
                                        memset(ld, 0, sizeof(minst_t));
                                        ld->op = AMD_SCRATCH_LOAD_DWORD;
                                        ld->num_defs = 1;
                                        ld->num_uses = 2;
                                        ld->operands[0].kind = MOP_VREG_V;
                                        ld->operands[0].reg_num = (uint16_t)new_vr;
                                        ld->operands[1].kind = MOP_IMM;
                                        ld->operands[1].imm = 0;
                                        ld->operands[2].kind = MOP_IMM;
                                        ld->operands[2].imm = (int32_t)spill_off[si];
                                    }
                                }
                            }
                            skip_reload:;
                        }
                    }

                    if (out_count < out_cap)
                        ra_output[out_count++] = orig;

                    for (uint32_t si = 0; si < num_spills; si++) {
                        uint32_t sv = spill_vreg[si];
                        int has_def = 0;
                        for (uint8_t d = 0; d < orig.num_defs && d < MINST_MAX_OPS; d++) {
                            if (operand_vreg(&orig.operands[d]) == sv) has_def = 1;
                        }
                        if (has_def) {
                            if (spill_file[si] == 0) {
                                /* SGPR spill store: move SGPR into a temp
                                   VGPR, then scratch_store from it.
                                   v_mov_b32 vTmp, sDef
                                   scratch_store_dword vOff, vTmp, spill_off */
                                uint32_t vtmp = A->vreg_count;
                                if (vtmp >= AMD_MAX_VREGS) continue;
                                A->reg_file[vtmp] = 1; /* VGPR */
                                A->vreg_count++;

                                /* v_mov_b32 vTmp, sDef */
                                if (out_count < out_cap) {
                                    minst_t *mv = &ra_output[out_count++];
                                    memset(mv, 0, sizeof(minst_t));
                                    mv->op = AMD_V_MOV_B32;
                                    mv->num_defs = 1;
                                    mv->num_uses = 1;
                                    mv->operands[0].kind = MOP_VREG_V;
                                    mv->operands[0].reg_num = (uint16_t)vtmp;
                                    mv->operands[1].kind = MOP_VREG_S;
                                    mv->operands[1].reg_num = (uint16_t)sv;
                                }
                                /* scratch_store_dword vOff, vTmp, off */
                                if (out_count < out_cap) {
                                    minst_t *st = &ra_output[out_count++];
                                    memset(st, 0, sizeof(minst_t));
                                    st->op = AMD_SCRATCH_STORE_DWORD;
                                    st->num_defs = 0;
                                    st->num_uses = 3;
                                    st->operands[0].kind = MOP_IMM;
                                    st->operands[0].imm = 0;
                                    st->operands[1].kind = MOP_VREG_V;
                                    st->operands[1].reg_num = (uint16_t)vtmp;
                                    st->operands[2].kind = MOP_IMM;
                                    st->operands[2].imm = (int32_t)spill_off[si];
                                }
                            } else {
                                if (out_count < out_cap) {
                                    minst_t *st = &ra_output[out_count++];
                                    memset(st, 0, sizeof(minst_t));
                                    st->op = AMD_SCRATCH_STORE_DWORD;
                                    st->num_defs = 0;
                                    st->num_uses = 3;
                                    st->operands[0].kind = MOP_IMM;
                                    st->operands[0].imm = 0;
                                    st->operands[1].kind = MOP_VREG_V;
                                    st->operands[1].reg_num = (uint16_t)sv;
                                    st->operands[2].kind = MOP_IMM;
                                    st->operands[2].imm = (int32_t)spill_off[si];
                                }
                            }
                        }
                    }
                }

                blk_out_count[bi2] = out_count - blk_out_start[bi2];
            }

            /* Single bulk-copy: compute total delta, shift tail once,
               then copy all expanded blocks into the instruction array. */
            uint32_t old_total = 0;
            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++)
                old_total += A->mblocks[F->first_block + bi2].num_insts;

            int32_t total_delta = (int32_t)out_count - (int32_t)old_total;
            uint32_t func_end = A->mblocks[F->first_block].first_inst + old_total;
            uint32_t tail = A->num_minsts - func_end;

            if (total_delta != 0 &&
                A->num_minsts + (uint32_t)total_delta <= AMD_MAX_MINSTS) {
                if (tail > 0) {
                    memmove(&A->minsts[(uint32_t)((int32_t)func_end + total_delta)],
                            &A->minsts[func_end],
                            tail * sizeof(minst_t));
                }
                A->num_minsts = (uint32_t)((int32_t)A->num_minsts + total_delta);
            }

            uint32_t write_pos = A->mblocks[F->first_block].first_inst;
            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++) {
                mblock_t *MB2 = &A->mblocks[F->first_block + bi2];
                MB2->first_inst = write_pos;
                MB2->num_insts = blk_out_count[bi2];
                memcpy(&A->minsts[write_pos],
                       &ra_output[blk_out_start[bi2]],
                       blk_out_count[bi2] * sizeof(minst_t));
                write_pos += blk_out_count[bi2];
            }

            for (uint32_t later = F->first_block + F->num_blocks;
                 later < A->num_mblocks; later++)
                A->mblocks[later].first_inst =
                    (uint32_t)((int32_t)A->mblocks[later].first_inst + total_delta);

            free(ra_output);
            free(blk_out_start);
            free(blk_out_count);

            free(ra_liveness_mem); ra_liveness_mem = NULL;
            free(ra_mem); ra_mem = NULL;
            continue;
        }

        /* --- Write reg_map from coloring --- */
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            uint32_t vreg = ra_nodes[n].vreg;
            uint16_t color = ra_nodes[n].color;
            if (color != 0xFFFF && vreg < AMD_MAX_VREGS)
                A->reg_map[vreg] = color;
            else if (vreg < AMD_MAX_VREGS)
                A->reg_map[vreg] = 0;
        }

        F->num_sgprs = max_sgpr;
        F->num_vgprs = max_vgpr;
        free(ra_liveness_mem); ra_liveness_mem = NULL;
        free(ra_mem); ra_mem = NULL;
        gc_success = 1;
        break; /* success, no spills */
    }

    if (!gc_success) {
        /* Spill iterations exhausted --fall back to linear scan */
        regalloc_linear(A, mf_idx);
        return;
    }

    finalize_reg_counts(A, F);
    rewrite_operands(A, F);
    dead_copy_elim(A, F);
}

/* Global flag: set by --no-graphcolor to force linear scan */
int amdgpu_force_linear_scan = 0;
/* If non-zero, cap available VGPRs for regalloc (forces spills for testing) */
int amdgpu_max_vgprs = 0;

static void regalloc_function(amd_module_t *A, uint32_t mf_idx)
{
    if (amdgpu_force_linear_scan || A->vreg_count > RA_MAX_NODES) {
        regalloc_linear(A, mf_idx);
    } else {
        regalloc_graphcolor(A, mf_idx);
    }
}

/* ---- Assembly Text Printer ---- */

static void asm_append(amd_module_t *A, const char *fmt, ...)
{
    if (A->asm_len >= AMD_ASM_SIZE - 256) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(A->asm_buf + A->asm_len, AMD_ASM_SIZE - A->asm_len, fmt, ap);
    va_end(ap);
    if (n > 0) A->asm_len += (uint32_t)n;
}

static void print_operand(amd_module_t *A, const moperand_t *op)
{
    switch (op->kind) {
    case MOP_SGPR:
        asm_append(A, "s%u", op->reg_num);
        break;
    case MOP_VGPR:
        asm_append(A, "v%u", op->reg_num);
        break;
    case MOP_VREG_S:
        asm_append(A, "%%vs%u", op->reg_num);
        break;
    case MOP_VREG_V:
        asm_append(A, "%%vv%u", op->reg_num);
        break;
    case MOP_IMM:
        /* Print hex for large values, decimal for small */
        if (op->imm >= -16 && op->imm <= 64)
            asm_append(A, "%d", op->imm);
        else
            asm_append(A, "0x%x", (uint32_t)op->imm);
        break;
    case MOP_LABEL:
        asm_append(A, ".LBB%u", (uint32_t)op->imm);
        break;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:
            asm_append(A, A->target <= AMD_TARGET_GFX950 ? "vcc" : "vcc_lo");
            break;
        case AMD_SPEC_EXEC:
            asm_append(A, A->target <= AMD_TARGET_GFX950 ? "exec" : "exec_lo");
            break;
        case AMD_SPEC_SCC:  asm_append(A, "scc"); break;
        case AMD_SPEC_M0:   asm_append(A, "m0"); break;
        default:            asm_append(A, "???"); break;
        }
        break;
    default:
        break;
    }
}

static void print_sgpr_pair(amd_module_t *A, uint16_t base)
{
    asm_append(A, "s[%u:%u]", base, base + 1);
}

static void print_minst(amd_module_t *A, const minst_t *mi)
{
    if (mi->op >= AMD_OP_COUNT) return;
    const amd_enc_entry_t *tbl = get_enc_table(A);
    const amd_enc_entry_t *enc = &tbl[mi->op];
    if (enc->mnemonic == NULL) return;

    /* Skip pseudo-instructions that survived */
    if (enc->fmt == AMD_FMT_PSEUDO) return;

    asm_append(A, "    %s", enc->mnemonic);

    /* Format-specific operand printing */
    uint8_t total = mi->num_defs + mi->num_uses;
    if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

    switch (enc->fmt) {
    case AMD_FMT_SMEM: {
        /* s_load_dword[x2/x4] sDst, sBase, offset */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            if (mi->op == AMD_S_LOAD_DWORDX2 || mi->op == AMD_S_LOAD_DWORDX4) {
                uint16_t base = mi->operands[0].reg_num;
                uint16_t cnt = (mi->op == AMD_S_LOAD_DWORDX2) ? 2 : 4;
                asm_append(A, "s[%u:%u]", base, base + cnt - 1);
            } else {
                print_operand(A, &mi->operands[0]);
            }
            asm_append(A, ", ");
            /* Base is a pair */
            if (mi->operands[1].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[1].reg_num);
            else
                print_operand(A, &mi->operands[1]);
            if (mi->num_uses > 1) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[2]);
            }
        }
        break;
    }
    case AMD_FMT_SOPP: {
        /* s_branch target / s_waitcnt encoding / s_endpgm / s_barrier */
        if (mi->op == AMD_S_WAITCNT) {
            uint16_t w = mi->flags;
            int vm = (w & AMD_WAIT_VMCNT0) != 0;
            int lgkm = (w & AMD_WAIT_LGKMCNT0) != 0;
            if (vm && lgkm)
                asm_append(A, " vmcnt(0) lgkmcnt(0)");
            else if (vm)
                asm_append(A, " vmcnt(0)");
            else if (lgkm)
                asm_append(A, " lgkmcnt(0)");
            else
                asm_append(A, " 0x%04x", w);
        } else if (mi->op == AMD_S_WAIT_LOADCNT ||
                   mi->op == AMD_S_WAIT_STORECNT ||
                   mi->op == AMD_S_WAIT_DSCNT ||
                   mi->op == AMD_S_WAIT_KMCNT) {
            asm_append(A, " 0x%x", mi->flags);
        } else if (mi->num_uses > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
        }
        break;
    }
    case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR: {
        /* Load:  global_load_dword  vDst, vOffset, sBase|off */
        /* Store: global_store_dword vOffset, vSrc, sBase|off */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            asm_append(A, ", ");
            print_operand(A, &mi->operands[1]);
            asm_append(A, ", ");
            if (mi->num_uses > 1 && mi->operands[2].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[2].reg_num);
            else
                asm_append(A, "off");
        } else {
            if (mi->num_uses >= 2) {
                asm_append(A, " ");
                print_operand(A, &mi->operands[0]);
                asm_append(A, ", ");
                print_operand(A, &mi->operands[1]);
                asm_append(A, ", ");
                if (mi->num_uses > 2 && mi->operands[2].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[2].reg_num);
                else
                    asm_append(A, "off");
            }
        }
        if (mi->flags & AMD_FLAG_GLC) asm_append(A, " glc");
        break;
    }
    case AMD_FMT_DS: {
        /* ds_read_b32 vDst, vAddr [, offset] */
        /* ds_write_b32 vAddr, vSrc [, offset] */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            for (uint8_t k = mi->num_defs; k < total; k++) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[k]);
            }
        } else {
            for (uint8_t k = 0; k < mi->num_uses && k < MINST_MAX_OPS; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    case AMD_FMT_SOP1: {
        /* s_mov_b32 sDst, sSrc */
        /* s_setpc_b64 sBase */
        if (mi->op == AMD_S_SETPC_B64 || mi->op == AMD_S_SWAPPC_B64) {
            if (mi->num_uses > 0) {
                asm_append(A, " ");
                if (mi->operands[mi->num_defs].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[mi->num_defs].reg_num);
                else
                    print_operand(A, &mi->operands[mi->num_defs]);
            }
        } else {
            for (uint8_t k = 0; k < total; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    default: {
        /* Generic: dst, src0, src1, ... */
        for (uint8_t k = 0; k < total; k++) {
            if (k > 0) asm_append(A, ",");
            asm_append(A, " ");
            print_operand(A, &mi->operands[k]);
        }
        break;
    }
    }

    asm_append(A, "\n");
}

static void emit_asm_function(amd_module_t *A, uint32_t mf_idx)
{
    const mfunc_t *F = &A->mfuncs[mf_idx];
    const char *name = A->bir->strings + F->name;

    if (F->is_kernel) {
        asm_append(A, "    .globl %s\n", name);
        asm_append(A, "    .p2align 8\n");
        asm_append(A, "    .type %s,@function\n", name);
    } else {
        asm_append(A, "    .type %s,@function\n", name);
    }
    asm_append(A, "%s:\n", name);
    asm_append(A, "    ; %u SGPRs, %u VGPRs, %u LDS bytes, %u scratch bytes\n",
               F->num_sgprs, F->num_vgprs, F->lds_bytes, F->scratch_bytes);

    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        const mblock_t *MB = &A->mblocks[mb_idx];

        asm_append(A, ".LBB%u:\n", mb_idx);

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            print_minst(A, &A->minsts[mi_idx]);
        }
    }
    asm_append(A, "\n");
}

void amdgpu_regalloc(amd_module_t *A)
{
    amdgpu_phi_elim(A);
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++)
        regalloc_function(A, fi);
}

void amdgpu_emit_asm(const amd_module_t *amd, FILE *out)
{
    /* We need to cast away const for the asm buffer operations */
    amd_module_t *A = (amd_module_t *)amd;

    A->asm_len = 0;
    asm_append(A, "    .amdgcn_target \"amdgcn-amd-amdhsa--%s\"\n",
               A->chip_name);
    asm_append(A, "    .text\n\n");

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        emit_asm_function(A, fi);
    }

    /* Write to output */
    fwrite(A->asm_buf, 1, A->asm_len, out);
}

/* ---- Msgpack Encoder (minimal, bounded) ---- */

#define MP_BUF_MAX 8192

static void mp_write(uint8_t *buf, uint32_t *pos, const void *data, uint32_t len)
{
    if (*pos + len > MP_BUF_MAX) return;
    memcpy(buf + *pos, data, len);
    *pos += len;
}

static void mp_fixmap(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x80 | count);
}

static void mp_fixarray(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x90 | count);
}

static void mp_fixstr(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint8_t len = (uint8_t)strlen(s);
    if (len > 31) len = 31;
    if (*pos + 1 + len > MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0xA0 | len);
    mp_write(buf, pos, s, len);
}

static void mp_str(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint32_t len = (uint32_t)strlen(s);
    if (len <= 31) {
        mp_fixstr(buf, pos, s);
    } else if (len <= 255) {
        if (*pos + 2 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xD9;
        buf[(*pos)++] = (uint8_t)len;
        mp_write(buf, pos, s, len);
    } else {
        if (*pos + 3 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xDA;
        buf[(*pos)++] = (uint8_t)(len >> 8);
        buf[(*pos)++] = (uint8_t)(len);
        mp_write(buf, pos, s, len);
    }
}

static void mp_uint(uint8_t *buf, uint32_t *pos, uint32_t val)
{
    if (*pos >= MP_BUF_MAX - 5) return;  /* worst case: 5 bytes */
    if (val <= 127) {
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFF) {
        buf[(*pos)++] = 0xCC;
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFFFF) {
        buf[(*pos)++] = 0xCD;
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    } else {
        buf[(*pos)++] = 0xCE;
        buf[(*pos)++] = (uint8_t)(val >> 24);
        buf[(*pos)++] = (uint8_t)(val >> 16);
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    }
}

static void mp_bool(uint8_t *buf, uint32_t *pos, int val)
{
    if (*pos >= MP_BUF_MAX - 1) return;
    buf[(*pos)++] = val ? 0xC3u : 0xC2u;
}

/* Emit one hidden-arg entry: fixmap(3) with .offset, .size, .value_kind */
static void mp_hidden_arg(uint8_t *buf, uint32_t *pos,
                          uint32_t offset, uint32_t size,
                          const char *value_kind)
{
    mp_fixmap(buf, pos, 3);
    mp_fixstr(buf, pos, ".offset");
    mp_uint(buf, pos, offset);
    mp_fixstr(buf, pos, ".size");
    mp_uint(buf, pos, size);
    mp_fixstr(buf, pos, ".value_kind");
    mp_fixstr(buf, pos, value_kind);
}

/* ---- ELF Code Object Writer ---- */

/* Local emit_dword for kernel descriptor padding (encode.c owns the
   encoding-side copy, but we need one here for the ELF layout too) */
static void emit_dword_elf(amd_module_t *A, uint32_t dw)
{
    if (A->code_len + 4 > AMD_CODE_SIZE) return;
    memcpy(A->code + A->code_len, &dw, 4);
    A->code_len += 4;
}

/* ELF64 types */
typedef struct {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} elf64_ehdr_t;   /* 64 bytes */

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} elf64_shdr_t;   /* 64 bytes */

typedef struct {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} elf64_sym_t;    /* 24 bytes */

typedef struct {
    uint32_t n_namesz;
    uint32_t n_descsz;
    uint32_t n_type;
} elf64_nhdr_t;   /* 12 bytes */

#define SHT_NULL     0
#define SHT_PROGBITS 1
#define SHT_SYMTAB   2
#define SHT_STRTAB   3
#define SHT_NOTE     7
#define SHF_ALLOC    2
#define SHF_EXECINSTR 4
#define STB_GLOBAL   1
#define STT_FUNC     2
#define STT_OBJECT   1
#define NT_AMDGPU_METADATA 32

/* Pad file to alignment boundary (bounded to avoid infinite loops) */
static void fwrite_pad(FILE *fp, uint32_t align)
{
    long pos = ftell(fp);
    if (pos < 0) return;
    uint32_t pad = ((uint32_t)pos + align - 1) & ~(align - 1);
    uint32_t n = pad - (uint32_t)pos;
    if (n > 256) n = 256; /* sanity cap */
    for (uint32_t i = 0; i < n; i++)
        fputc(0, fp);
}

int amdgpu_emit_elf(amd_module_t *A, const char *path)
{
    /* First, encode all functions to binary */
    A->code_len = 0;

    /* Track kernel descriptor positions */
    static uint32_t kd_offsets[AMD_MAX_MFUNCS];
    static uint32_t code_offsets[AMD_MAX_MFUNCS];
    uint32_t num_kernels = 0;

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;

        /* Align to 256 bytes for kernel descriptor */
        for (uint32_t pad = 0; A->code_len % 256 != 0 && pad < 64; pad++)
            emit_dword_elf(A, 0);
        kd_offsets[num_kernels] = A->code_len;

        /* Write kernel descriptor (64 bytes) */
        amd_kernel_descriptor_t kd;
        memset(&kd, 0, sizeof(kd));

        mfunc_t *F = &A->mfuncs[fi];
        kd.group_segment_fixed_size = F->lds_bytes;
        kd.private_segment_fixed_size = F->scratch_bytes;
        kd.kernarg_size = F->kernarg_bytes + 256; /* +256 for COV5 hidden args */
        kd.kernel_code_entry_byte_offset = 256; /* descriptor is 64 bytes, padded to 256 */

        /* compute_pgm_rsrc1 — VGPR granularity: 8 for CDNA, 8 for RDNA */
        int cdna = (A->target <= AMD_TARGET_GFX950);
        uint32_t vgran = cdna ? 8u : 8u;
        uint32_t vgpr_blocks = (F->num_vgprs > 0)
            ? (uint32_t)((F->num_vgprs + vgran - 1) / vgran - 1) : 0;
        uint32_t sgpr_blocks = (F->num_sgprs > 0) ? (uint32_t)((F->num_sgprs + 7) / 8 - 1) : 0;
        kd.compute_pgm_rsrc1 = (vgpr_blocks & 0x3F) |
                               ((sgpr_blocks & 0xF) << 6) |
                               (0xFu << 16) |  /* FLOAT_MODE: denorms enabled */
                               (1u << 21) |  /* DX10_CLAMP */
                               (1u << 23);   /* IEEE_MODE */
        if (!cdna) {
            kd.compute_pgm_rsrc1 |= (1u << 26) |  /* WGP_MODE (RDNA only) */
                                    (1u << 27);    /* MEM_ORDERED (RDNA only) */
        }

        /* compute_pgm_rsrc2 */
        kd.compute_pgm_rsrc2 = ((F->scratch_bytes > 0) ? 1u : 0u) | /* SCRATCH_EN */
                               (4u << 1) |             /* USER_SGPR_COUNT = 4 */
                               (1u << 7) |             /* ENABLE_SGPR_WORKGROUP_ID_X */
                               (1u << 8) |             /* ENABLE_SGPR_WORKGROUP_ID_Y */
                               (1u << 9);              /* ENABLE_SGPR_WORKGROUP_ID_Z */

        /* compute_pgm_rsrc3 — CDNA: set ACCUM_OFFSET so all VGPRs are arch */
        if (cdna) {
            /* arch VGPRs = (ACCUM_OFFSET+1)*4, total = (vgpr_blocks+1)*8.
             * Set ACCUM_OFFSET = (vgpr_blocks+1)*2 - 1 so accum starts past end. */
            uint32_t accum_off = (vgpr_blocks + 1u) * 2u - 1u;
            kd.compute_pgm_rsrc3 = accum_off & 0x3Fu;
        }

        /* kernel_code_properties */
        kd.kernel_code_properties = (1u << 1) |  /* ENABLE_SGPR_DISPATCH_PTR */
                                    (1u << 3);   /* ENABLE_SGPR_KERNARG_SEGMENT_PTR */

        /* Write the 64 bytes */
        if (A->code_len + 64 <= AMD_CODE_SIZE) {
            memcpy(A->code + A->code_len, &kd, 64);
            A->code_len += 64;
        }

        /* Pad to 256 bytes after descriptor */
        for (uint32_t pad = 0; A->code_len % 256 != 0 && A->code_len < AMD_CODE_SIZE && pad < 256; pad++)
            A->code[A->code_len++] = 0;

        code_offsets[num_kernels] = A->code_len;
        num_kernels++;

        /* Encode the function's instructions */
        encode_function(A, fi);
    }

    /* Also encode device functions */
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (A->mfuncs[fi].is_kernel) continue;
        encode_function(A, fi);
    }

    /* Build note section (msgpack metadata) */
    static uint8_t note_buf[16384];
    uint32_t note_len = 0;

    /* Note header: name = "AMDGPU", type = NT_AMDGPU_METADATA */
    const char *note_name = "AMDGPU\0\0"; /* 8 bytes aligned */
    uint32_t note_name_len = 7; /* including null */

    /* Build msgpack payload */
    static uint8_t mp_buf[8192];
    uint32_t mp_pos = 0;

    mp_fixmap(mp_buf, &mp_pos, 3);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.version");
    mp_fixarray(mp_buf, &mp_pos, 2);
    mp_uint(mp_buf, &mp_pos, 1);
    mp_uint(mp_buf, &mp_pos, 2);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.target");
    char mp_tgt[40];
    snprintf(mp_tgt, sizeof(mp_tgt), "amdgcn-amd-amdhsa--%s", A->chip_name);
    mp_str(mp_buf, &mp_pos, mp_tgt);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.kernels");
    uint8_t nk = (num_kernels > 15) ? 15 : (uint8_t)num_kernels;
    mp_fixarray(mp_buf, &mp_pos, nk);

    uint32_t ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < nk; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        mfunc_t *F = &A->mfuncs[fi];
        const char *name = A->bir->strings + F->name;

        /* Build symbol name: "name.kd" */
        char kd_name[256];
        snprintf(kd_name, sizeof(kd_name), "%s.kd", name);

        mp_fixmap(mp_buf, &mp_pos, 15);

        mp_fixstr(mp_buf, &mp_pos, ".name");
        mp_str(mp_buf, &mp_pos, name);

        mp_fixstr(mp_buf, &mp_pos, ".symbol");
        mp_str(mp_buf, &mp_pos, kd_name);

        mp_str(mp_buf, &mp_pos, ".kernarg_segment_size");
        mp_uint(mp_buf, &mp_pos, F->kernarg_bytes + 256); /* +256 for hidden args */

        mp_str(mp_buf, &mp_pos, ".kernarg_segment_align");
        mp_uint(mp_buf, &mp_pos, 8);

        mp_str(mp_buf, &mp_pos, ".group_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->lds_bytes);

        mp_str(mp_buf, &mp_pos, ".private_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->scratch_bytes);

        mp_str(mp_buf, &mp_pos, ".wavefront_size");
        mp_uint(mp_buf, &mp_pos, F->wavefront_size);

        mp_fixstr(mp_buf, &mp_pos, ".sgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_sgprs);

        mp_fixstr(mp_buf, &mp_pos, ".vgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_vgprs);

        mp_str(mp_buf, &mp_pos, ".agpr_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".sgpr_spill_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".vgpr_spill_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".max_flat_workgroup_size");
        mp_uint(mp_buf, &mp_pos, 1024);

        mp_str(mp_buf, &mp_pos, ".uses_dynamic_stack");
        mp_bool(mp_buf, &mp_pos, 0);

        mp_fixstr(mp_buf, &mp_pos, ".args");
        mp_fixarray(mp_buf, &mp_pos, 14); /* 1 user + 13 hidden args */
        {
            /* User args as a single by_value blob */
            mp_fixmap(mp_buf, &mp_pos, 3);
            mp_fixstr(mp_buf, &mp_pos, ".offset");
            mp_uint(mp_buf, &mp_pos, 0);
            mp_fixstr(mp_buf, &mp_pos, ".size");
            mp_uint(mp_buf, &mp_pos, F->kernarg_bytes);
            mp_fixstr(mp_buf, &mp_pos, ".value_kind");
            mp_fixstr(mp_buf, &mp_pos, "by_value");

            uint32_t ho = F->kernarg_bytes; /* hidden args start after user args */
            mp_hidden_arg(mp_buf, &mp_pos, ho +  0, 4, "hidden_block_count_x");
            mp_hidden_arg(mp_buf, &mp_pos, ho +  4, 4, "hidden_block_count_y");
            mp_hidden_arg(mp_buf, &mp_pos, ho +  8, 4, "hidden_block_count_z");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 12, 2, "hidden_group_size_x");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 14, 2, "hidden_group_size_y");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 16, 2, "hidden_group_size_z");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 18, 2, "hidden_remainder_x");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 20, 2, "hidden_remainder_y");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 22, 2, "hidden_remainder_z");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 24, 8, "hidden_global_offset_x");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 32, 8, "hidden_global_offset_y");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 40, 8, "hidden_global_offset_z");
            mp_hidden_arg(mp_buf, &mp_pos, ho + 48, 2, "hidden_grid_dims");
        }

        ki++;
    }

    /* Assemble note section */
    elf64_nhdr_t nhdr;
    nhdr.n_namesz = note_name_len;
    nhdr.n_descsz = mp_pos;
    nhdr.n_type = NT_AMDGPU_METADATA;
    memcpy(note_buf + note_len, &nhdr, 12);
    note_len += 12;
    memcpy(note_buf + note_len, note_name, 8); /* padded to 4-byte align */
    note_len += 8;
    memcpy(note_buf + note_len, mp_buf, mp_pos);
    note_len += mp_pos;
    /* Pad to 4 bytes */
    while (note_len % 4 != 0 && note_len < sizeof(note_buf))
        note_buf[note_len++] = 0;

    /* Build string tables */
    /* .shstrtab: section names (total < 50 bytes, 256 is generous) */
    #define SHSTRTAB_MAX 256
    static char shstrtab[SHSTRTAB_MAX];
    uint32_t shstrtab_len = 0;
    shstrtab[shstrtab_len++] = '\0';
    uint32_t text_name_off = shstrtab_len;
    if (shstrtab_len + 6 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".text", 6); shstrtab_len += 6; }
    uint32_t note_name_off = shstrtab_len;
    if (shstrtab_len + 6 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".note", 6); shstrtab_len += 6; }
    uint32_t symtab_name_off = shstrtab_len;
    if (shstrtab_len + 8 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".symtab", 8); shstrtab_len += 8; }
    uint32_t strtab_name_off = shstrtab_len;
    if (shstrtab_len + 8 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".strtab", 8); shstrtab_len += 8; }
    uint32_t shstrtab_name_off = shstrtab_len;
    if (shstrtab_len + 10 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".shstrtab", 10); shstrtab_len += 10; }

    /* .strtab: symbol names */
    #define STRTAB_MAX 4096
    static char strtab[STRTAB_MAX];
    uint32_t strtab_len = 0;
    strtab[strtab_len++] = '\0';

    /* Build symbol table */
    static elf64_sym_t symtab[256];
    uint32_t num_syms = 1; /* null symbol at index 0 */
    memset(&symtab[0], 0, sizeof(elf64_sym_t));

    ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < num_kernels; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        const char *name = A->bir->strings + A->mfuncs[fi].name;

        /* Kernel descriptor symbol (STT_OBJECT) */
        char kd_sym[256];
        snprintf(kd_sym, sizeof(kd_sym), "%s.kd", name);
        uint32_t kd_name_idx = strtab_len;
        uint32_t kd_sym_len = (uint32_t)strlen(kd_sym) + 1;
        if (strtab_len + kd_sym_len <= STRTAB_MAX) {
            memcpy(strtab + strtab_len, kd_sym, kd_sym_len);
            strtab_len += kd_sym_len;
        }

        if (num_syms < 256) {
            symtab[num_syms].st_name = kd_name_idx;
            symtab[num_syms].st_info = (STB_GLOBAL << 4) | STT_OBJECT;
            symtab[num_syms].st_other = 0;
            symtab[num_syms].st_shndx = 1; /* .text section */
            symtab[num_syms].st_value = kd_offsets[ki];
            symtab[num_syms].st_size = 64;
            num_syms++;
        }

        /* Function symbol (STT_FUNC) */
        uint32_t func_name_idx = strtab_len;
        uint32_t name_len = (uint32_t)strlen(name) + 1;
        if (strtab_len + name_len <= STRTAB_MAX) {
            memcpy(strtab + strtab_len, name, name_len);
            strtab_len += name_len;
        }

        if (num_syms < 256) {
            symtab[num_syms].st_name = func_name_idx;
            symtab[num_syms].st_info = (STB_GLOBAL << 4) | STT_FUNC;
            symtab[num_syms].st_other = 0;
            symtab[num_syms].st_shndx = 1;
            symtab[num_syms].st_value = code_offsets[ki];
            symtab[num_syms].st_size = A->code_len - code_offsets[ki];
            num_syms++;
        }

        ki++;
    }

    /* Write ELF file */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    /* Compute layout */
    uint64_t text_off = 64; /* after ELF header */
    uint64_t text_size = A->code_len;
    uint64_t note_off = text_off + text_size;
    /* Align note to 4 bytes */
    note_off = (note_off + 3) & ~3ULL;
    uint64_t symtab_off = note_off + note_len;
    symtab_off = (symtab_off + 7) & ~7ULL;
    uint64_t symtab_size = num_syms * 24;
    uint64_t strtab_off = symtab_off + symtab_size;
    uint64_t shstrtab_off = strtab_off + strtab_len;
    uint64_t shdr_off = (shstrtab_off + shstrtab_len + 7) & ~7ULL;

    /* Sections: 0=NULL, 1=.text, 2=.note, 3=.symtab, 4=.strtab, 5=.shstrtab */
    uint16_t num_sections = 6;

    /* ELF header */
    elf64_ehdr_t ehdr;
    memset(&ehdr, 0, sizeof(ehdr));
    ehdr.e_ident[0] = 0x7F;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = 2;    /* ELFCLASS64 */
    ehdr.e_ident[5] = 1;    /* ELFDATA2LSB */
    ehdr.e_ident[6] = 1;    /* EV_CURRENT */
    ehdr.e_ident[7] = ELFOSABI_AMDGPU_HSA;
    ehdr.e_ident[8] = 4;    /* ABI version: code object V4 */
    ehdr.e_type = 1;        /* ET_REL (required for code object V3+) */
    ehdr.e_machine = EM_AMDGPU;
    ehdr.e_version = 1;
    ehdr.e_entry = 0;
    ehdr.e_phoff = 0;
    ehdr.e_shoff = shdr_off;
    ehdr.e_flags = A->elf_mach;
    /* V4 feature flags: XNACK_ANY (bits[9:8]=01) + SRAMECC_ANY (bits[11:10]=01) */
    if (A->target <= AMD_TARGET_GFX950)
        ehdr.e_flags |= 0x500;  /* 0x100 | 0x400 */
    ehdr.e_ehsize = 64;
    ehdr.e_phentsize = 0;
    ehdr.e_phnum = 0;
    ehdr.e_shentsize = 64;
    ehdr.e_shnum = num_sections;
    ehdr.e_shstrndx = 5;    /* .shstrtab index */
    fwrite(&ehdr, 1, 64, fp);

    /* .text section */
    fwrite(A->code, 1, A->code_len, fp);
    fwrite_pad(fp, 4);

    /* .note section */
    fwrite(note_buf, 1, note_len, fp);
    fwrite_pad(fp, 8);

    /* .symtab */
    fwrite(symtab, 24, num_syms, fp);

    /* .strtab */
    fwrite(strtab, 1, strtab_len, fp);

    /* .shstrtab */
    fwrite(shstrtab, 1, shstrtab_len, fp);
    fwrite_pad(fp, 8);

    /* Section header table */
    elf64_shdr_t shdrs[6];
    memset(shdrs, 0, sizeof(shdrs));

    /* 0: NULL */
    /* 1: .text */
    shdrs[1].sh_name = text_name_off;
    shdrs[1].sh_type = SHT_PROGBITS;
    shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    shdrs[1].sh_offset = text_off;
    shdrs[1].sh_size = text_size;
    shdrs[1].sh_addralign = 256;

    /* 2: .note */
    shdrs[2].sh_name = note_name_off;
    shdrs[2].sh_type = SHT_NOTE;
    shdrs[2].sh_flags = SHF_ALLOC;
    shdrs[2].sh_offset = note_off;
    shdrs[2].sh_size = note_len;
    shdrs[2].sh_addralign = 4;

    /* 3: .symtab */
    shdrs[3].sh_name = symtab_name_off;
    shdrs[3].sh_type = SHT_SYMTAB;
    shdrs[3].sh_offset = symtab_off;
    shdrs[3].sh_size = symtab_size;
    shdrs[3].sh_link = 4;    /* .strtab */
    shdrs[3].sh_info = 1;    /* first non-local symbol */
    shdrs[3].sh_addralign = 8;
    shdrs[3].sh_entsize = 24;

    /* 4: .strtab */
    shdrs[4].sh_name = strtab_name_off;
    shdrs[4].sh_type = SHT_STRTAB;
    shdrs[4].sh_offset = strtab_off;
    shdrs[4].sh_size = strtab_len;
    shdrs[4].sh_addralign = 1;

    /* 5: .shstrtab */
    shdrs[5].sh_name = shstrtab_name_off;
    shdrs[5].sh_type = SHT_STRTAB;
    shdrs[5].sh_offset = shstrtab_off;
    shdrs[5].sh_size = shstrtab_len;
    shdrs[5].sh_addralign = 1;

    fwrite(shdrs, 64, num_sections, fp);

    fclose(fp);

    fprintf(stderr, "wrote %s (%u bytes code, %u kernels)\n",
            path, A->code_len, num_kernels);
    return BC_OK;
}
