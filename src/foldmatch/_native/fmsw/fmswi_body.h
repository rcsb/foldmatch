/*
 * fmswi_body.h -- one instance of the integer striped Smith-Waterman of fmsw.c (DP id fmswi/1, API and model in
 * fmsw.h). Not a public header: fmsw.c includes it twice, for 16-bit and 32-bit lanes, after defining
 *   SWI_T     lane type (int16_t | int32_t)          SWI_V     vector of SWI_T
 *   SWI_L     lanes per vector                        SWI_VB    uint8 vector of SWI_L lanes (one flag byte per lane)
 *   SWI_TMIN  most negative SWI_T (profile padding)   SWI_MAX(a, b), SWI_ANY(mask)
 *   SWI_SH1(v, x)  lane l <- lane l - 1, lane 0 <- x  SWI_FN(name)  name with the instance suffix
 *
 * Farrar striped layout: query row i = l * seg + k sits in lane l of segment k (seg = ceil(n / SWI_L)); rows >= n are
 * padding scored SWI_TMIN, so they never become positive and never feed a real row.
 */

/* profile [A][seg]: lane l of segment k = qprof[i][a] for query row i = l * seg + k; padding rows = SWI_TMIN */
static void SWI_FN(swi_profile)(SWI_V *P, const int32_t *qprof, int n, int A, int seg)
{
    for (int a = 0; a < A; a++)
        for (int k = 0; k < seg; k++) {
            SWI_V v;
            for (int l = 0; l < SWI_L; l++) {
                const int i = l * seg + k;
                v[l] = i < n ? (SWI_T)qprof[(size_t)i * (size_t)A + (size_t)a] : (SWI_T)SWI_TMIN;
            }
            P[(size_t)a * (size_t)seg + (size_t)k] = v;
        }
}

/*
 * One pair. Returns the best score; *overflow is set when it reached lim (the 16-bit pass is then redone in 32 bits:
 * from the next column on a lane could wrap). With tb != 0 the flags of every cell are written to fl, column-major in
 * the striped layout: fl[((j - 1) * seg + k) * SWI_L + l] for query row l * seg + k (bits: SWI_SRC_* = predecessor
 * state of M, SWI_DEAD = M <= 0 before the clip, SWI_G1X / SWI_G2X = G1 / G2 extends).
 *
 * Bounds that keep every lane in range (fmswi_align checks them before choosing 16 bits): G1, G2 >= go always (an
 * open from M >= 0), so g + ge >= go + ge > SWI_TMIN; M at column j <= best(j - 1) + smax < lim + smax; the lazy-F
 * carry is clamped at go before each extension (a carried value below go can never beat a G2 >= go, so the clamp
 * changes no result and keeps a long carry from wrapping).
 */
static int SWI_FN(swi_sw)(const SWI_V *P, int seg, const uint8_t *t, int m, SWI_T go, SWI_T ge, int tb, SWI_V *ws,
                          uint8_t *fl, int lim, int *overflow, int *bi_out, int *bj_out)
{
    SWI_V *H = ws, *G1 = ws + seg, *G2 = ws + 2 * (size_t)seg, *M = ws + 3 * (size_t)seg;
    const SWI_V zero = {0}, vgo = zero + go, vge = zero + ge;
    for (int k = 0; k < seg; k++) { H[k] = zero; G1[k] = vgo; G2[k] = vgo; M[k] = zero; }
    int best = 0, bi = 0, bj = 0;
    for (int j = 1; j <= m; j++) {
        const SWI_V *vP = P + (size_t)t[j - 1] * (size_t)seg;
        SWI_V hd = SWI_SH1(H[seg - 1], 0);
        SWI_V colmax = zero;
        SWI_V vF = vgo;
        SWI_V fextl = zero;                        /* trace: 'extends' mask of the final vF (into the next row) */
        if (!tb) {
            /* score only: G1[k] holds G1 of THIS column (computed while processing the previous one) */
            for (int k = 0; k < seg; k++) {
                const SWI_V hl = H[k];
                const SWI_V mc = SWI_MAX(hd + vP[k], zero);
                const SWI_V g1 = G1[k];
                const SWI_V mo = mc + vgo;
                G1[k] = SWI_MAX(mo, g1 + vge);
                H[k] = SWI_MAX(mc, SWI_MAX(g1, vF));
                G2[k] = vF;
                colmax = SWI_MAX(colmax, mc);
                vF = SWI_MAX(mo, vF + vge);
                hd = hl;
            }
        } else {
            uint8_t *fc = fl + (size_t)(j - 1) * (size_t)seg * SWI_L;
            SWI_V md = SWI_SH1(M[seg - 1], 0), g1d = SWI_SH1(G1[seg - 1], go), g2d = SWI_SH1(G2[seg - 1], go);
            SWI_V fext = zero;                     /* G2 'extends' mask of this segment (from the previous one) */
            for (int k = 0; k < seg; k++) {
                const SWI_V hl = H[k], mp = M[k], g1p = G1[k], g2p = G2[k];
                const SWI_V mraw = hd + vP[k];
                const SWI_V mc = SWI_MAX(mraw, zero);
                const SWI_V g1o = mp + vgo, g1e = g1p + vge;
                const SWI_V g1 = SWI_MAX(g1o, g1e);
                /* predecessor of M, biotite order M > G1 > G2 on ties (dead G <= 0 <= M never wins) */
                const SWI_V notm = (SWI_V)((g1d > md) | (g2d > md));
                const SWI_V srcg2 = notm & (SWI_V)(g2d > g1d);
                const SWI_V srcg1 = notm & ~srcg2;
                const SWI_V f = (srcg1 & SWI_SRC_G1) | (srcg2 & SWI_SRC_G2) | ((SWI_V)(mraw <= zero) & SWI_DEAD)
                                | ((SWI_V)(g1e > g1o) & SWI_G1X) | (fext & SWI_G2X);
                const SWI_VB fb = __builtin_convertvector(f, SWI_VB);
                memcpy(fc + (size_t)k * SWI_L, &fb, SWI_L);
                M[k] = mc; G1[k] = g1; G2[k] = vF;
                H[k] = SWI_MAX(mc, SWI_MAX(g1, vF));
                colmax = SWI_MAX(colmax, mc);
                const SWI_V mo = mc + vgo, fe = vF + vge;
                fext = (SWI_V)(fe > mo);
                vF = SWI_MAX(mo, fe);
                hd = hl; md = mp; g1d = g1p; g2d = g2p;
            }
            fextl = fext;
        }
        /* lazy F: carry G2 across the segment boundary while it improves any lane. The first carried vector mixes
           'open' (M of the row above + go) and 'extend' candidates (fx: the lanes that extend); every later
           improvement is an extension. */
        vF = SWI_SH1(vF, go);
        SWI_V fx = tb ? SWI_SH1(fextl, 0) : zero;
        for (int k = 0; SWI_ANY(vF > G2[k]);) {
            if (tb) {
                const SWI_V up = (SWI_V)(vF > G2[k]);
                uint8_t *fk = fl + ((size_t)(j - 1) * (size_t)seg + (size_t)k) * SWI_L;
                SWI_VB fb;
                memcpy(&fb, fk, SWI_L);
                fb |= __builtin_convertvector(up & fx & SWI_G2X, SWI_VB);
                memcpy(fk, &fb, SWI_L);
                fx = ~zero;
            }
            G2[k] = SWI_MAX(G2[k], vF);
            H[k] = SWI_MAX(H[k], vF);
            vF = SWI_MAX(vF, vgo) + vge;
            if (++k == seg) { k = 0; vF = SWI_SH1(vF, go); }
        }
        /* end cell: the first cell in row-major order with M == best. With gap_open >= 1 every G is below the best
           M seen so far, so H == best <=> M == best in a column that reaches the best. */
        const SWI_V vbest = zero + (SWI_T)(best > 0 ? best : 1);
        if (SWI_ANY(colmax >= vbest)) {
            int cm = 0;
            for (int l = 0; l < SWI_L; l++) cm = colmax[l] > cm ? colmax[l] : cm;
            if (cm >= best) {
                const SWI_V vc = zero + (SWI_T)cm;
                int imin = INT32_MAX;
                for (int k = 0; k < seg; k++) {
                    const SWI_V eq = (SWI_V)(H[k] == vc);
                    if (!SWI_ANY(eq)) continue;
                    for (int l = 0; l < SWI_L; l++)
                        if (eq[l] && l * seg + k < imin) imin = l * seg + k;
                }
                if (cm > best || imin + 1 < bi) { best = cm; bi = imin + 1; bj = j; }
            }
            if (best >= lim) break;                /* 16 bits could wrap from the next column on: redo in 32 bits */
        }
    }
    *overflow = best >= lim;
    *bi_out = bi; *bj_out = bj;
    return best;
}
