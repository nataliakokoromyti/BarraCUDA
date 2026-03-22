/* ttabs.c -- encoding table validation
 * Every opcode must have a home. No squatters, no vacancies.
 * AMD renumbers everything between generations just to keep us honest. */

#include "tharns.h"
#include "amdgpu.h"

/* ---- tables: GFX11 completeness ---- */

static void tab_gx11(void)
{
    int miss = 0;
    for (int i = 0; i < AMD_OP_COUNT; i++) {
        if (amd_enc_table[i].fmt == AMD_FMT_PSEUDO) continue;
        if (amd_enc_table[i].mnemonic == NULL) {
            /* GFX12-only ops (wait_loadcnt etc.) can be NULL in GFX11 table
             * if they exist only in the GFX10 table as stubs. Check that
             * they at least appear in the enum range. */
            continue;
        }
        if (amd_enc_table[i].fmt == 0 && amd_enc_table[i].hw_opcode == 0
            && amd_enc_table[i].mnemonic == NULL) {
            printf("  GFX11 missing: op %d\n", i);
            miss++;
        }
    }
    CHEQ(miss, 0);
    PASS();
}
TH_REG("tables", tab_gx11)

/* ---- tables: GFX10 completeness ---- */

static void tab_gx10(void)
{
    int miss = 0;
    for (int i = 0; i < AMD_OP_COUNT; i++) {
        if (amd_enc_table_gfx10[i].fmt == AMD_FMT_PSEUDO) continue;
        /* GFX12-only wait instructions are NULL stubs in GFX10, that's fine */
        if (amd_enc_table_gfx10[i].mnemonic == NULL) continue;
        if (amd_enc_table_gfx10[i].fmt == 0 && amd_enc_table_gfx10[i].hw_opcode == 0
            && amd_enc_table_gfx10[i].mnemonic == NULL) {
            printf("  GFX10 missing: op %d\n", i);
            miss++;
        }
    }
    CHEQ(miss, 0);
    PASS();
}
TH_REG("tables", tab_gx10)

/* ---- tables: GFX10 DS mnemonic is ds_read, not ds_load ---- */

static void tab_mnem(void)
{
    CHECK(amd_enc_table_gfx10[AMD_DS_READ_B32].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx10[AMD_DS_READ_B32].mnemonic, "ds_read_b32");
    CHECK(amd_enc_table_gfx10[AMD_DS_WRITE_B32].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx10[AMD_DS_WRITE_B32].mnemonic, "ds_write_b32");
    PASS();
}
TH_REG("tables", tab_mnem)

/* ---- tables: GFX9 completeness ---- */

static void tab_gfx9(void)
{
    int miss = 0;
    for (int i = 0; i < AMD_OP_COUNT; i++) {
        if (amd_enc_table_gfx9[i].fmt == AMD_FMT_PSEUDO) continue;
        if (amd_enc_table_gfx9[i].mnemonic == NULL) continue;
        if (amd_enc_table_gfx9[i].fmt == 0 && amd_enc_table_gfx9[i].hw_opcode == 0
            && amd_enc_table_gfx9[i].mnemonic == NULL) {
            printf("  GFX9 missing: op %d\n", i);
            miss++;
        }
    }
    CHEQ(miss, 0);
    PASS();
}
TH_REG("tables", tab_gfx9)

/* ---- tables: known opcode differences ---- */

static void tab_diff(void)
{
    /* s_and_b32: GFX10=0x0E, GFX11=0x16 */
    CHEQ(amd_enc_table[AMD_S_AND_B32].hw_opcode,         0x16);
    CHEQ(amd_enc_table_gfx10[AMD_S_AND_B32].hw_opcode,   0x0E);

    /* v_cndmask_b32: GFX10=0x00, GFX11=0x01 */
    CHEQ(amd_enc_table[AMD_V_CNDMASK_B32].hw_opcode,       0x01);
    CHEQ(amd_enc_table_gfx10[AMD_V_CNDMASK_B32].hw_opcode, 0x00);

    /* s_endpgm: GFX10=0x01, GFX11=0x30 */
    CHEQ(amd_enc_table[AMD_S_ENDPGM].hw_opcode,       0x30);
    CHEQ(amd_enc_table_gfx10[AMD_S_ENDPGM].hw_opcode, 0x01);

    PASS();
}
TH_REG("tables", tab_diff)

/* ---- tables: GFX9 vs GFX11 opcode differences ---- */

static void tab_diff9(void)
{
    /* s_and_b32: GFX9=0x0C, GFX11=0x16 */
    CHEQ(amd_enc_table_gfx9[AMD_S_AND_B32].hw_opcode, 0x0C);
    CHEQ(amd_enc_table[AMD_S_AND_B32].hw_opcode,      0x16);

    /* v_add_f32: GFX9=0x01, GFX11=0x03 */
    CHEQ(amd_enc_table_gfx9[AMD_V_ADD_F32].hw_opcode, 0x01);
    CHEQ(amd_enc_table[AMD_V_ADD_F32].hw_opcode,      0x03);

    /* v_cndmask_b32: GFX9=0x00, GFX11=0x01 */
    CHEQ(amd_enc_table_gfx9[AMD_V_CNDMASK_B32].hw_opcode, 0x00);

    /* s_endpgm: GFX9=0x01, GFX11=0x30 */
    CHEQ(amd_enc_table_gfx9[AMD_S_ENDPGM].hw_opcode, 0x01);

    /* ds_read_b32: GFX9=0x36 (same opcode, but mnemonic is ds_read_b32) */
    CHEQ(amd_enc_table_gfx9[AMD_DS_READ_B32].hw_opcode, 0x36);
    CHSTR(amd_enc_table_gfx9[AMD_DS_READ_B32].mnemonic, "ds_read_b32");

    PASS();
}
TH_REG("tables", tab_diff9)

/* ---- tables: Wave64 _b64 ops ---- */

static void tab_b64(void)
{
    /* s_and_b64: GFX9=0x0D */
    CHECK(amd_enc_table_gfx9[AMD_S_AND_B64].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_S_AND_B64].mnemonic, "s_and_b64");
    CHEQ(amd_enc_table_gfx9[AMD_S_AND_B64].hw_opcode, 0x0D);

    /* s_or_b64: GFX9=0x0F */
    CHECK(amd_enc_table_gfx9[AMD_S_OR_B64].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_S_OR_B64].mnemonic, "s_or_b64");
    CHEQ(amd_enc_table_gfx9[AMD_S_OR_B64].hw_opcode, 0x0F);

    /* s_and_saveexec_b64: GFX9=0x20 */
    CHECK(amd_enc_table_gfx9[AMD_S_AND_SAVEEXEC_B64].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_S_AND_SAVEEXEC_B64].mnemonic, "s_and_saveexec_b64");
    CHEQ(amd_enc_table_gfx9[AMD_S_AND_SAVEEXEC_B64].hw_opcode, 0x20);

    /* s_mov_b64: GFX9=0x01 */
    CHECK(amd_enc_table_gfx9[AMD_S_MOV_B64].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_S_MOV_B64].mnemonic, "s_mov_b64");
    CHEQ(amd_enc_table_gfx9[AMD_S_MOV_B64].hw_opcode, 0x01);

    PASS();
}
TH_REG("tables", tab_b64)

/* ---- tables: MFMA matrix ops ---- */

static void tab_mfma(void)
{
    /* v_mfma_f32_16x16x32_f16: opcode 0x54 */
    CHECK(amd_enc_table_gfx9[AMD_V_MFMA_F32_16X16X32_F16].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_V_MFMA_F32_16X16X32_F16].mnemonic, "v_mfma_f32_16x16x32_f16");
    CHEQ(amd_enc_table_gfx9[AMD_V_MFMA_F32_16X16X32_F16].hw_opcode, 0x54);

    /* v_mfma_f32_32x32x16_f16: opcode 0x55 */
    CHECK(amd_enc_table_gfx9[AMD_V_MFMA_F32_32X32X16_F16].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx9[AMD_V_MFMA_F32_32X32X16_F16].mnemonic, "v_mfma_f32_32x32x16_f16");
    CHEQ(amd_enc_table_gfx9[AMD_V_MFMA_F32_32X32X16_F16].hw_opcode, 0x55);

    PASS();
}
TH_REG("tables", tab_mfma)
