// Kalman Filter

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "libmat.h"
#include "structs.h"
#include "defines.h"
#include "matrices.h"


void ekf_p(struct wm *wm, struct ss_d *ss) {
        // Declaration of the variables
        struct matrix *x, *xp, *Phi, *Q, *P, *Pp;
        struct matrix *PhiP, *Phi_T, *PhiPPhi_T;
        int nrPhi, ncPhi, ncP;

        x  = ss->x;
        xp = ss->xp;
        Phi = ss->Phi;
        Q  = wm->Q;
        P = wm->P;
        Pp = wm->Pp;

        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;
        ncP = wm->P->n_col;

        PhiP = matrix_m(nrPhi, ncP);
        Phi_T = matrix_m(ncPhi, nrPhi);
        PhiPPhi_T = matrix_m(nrPhi, nrPhi);

        times_m(Phi,x,xp);
        times_m(Phi,P,PhiP);
        transp_m(Phi,Phi_T);
        times_m (PhiP, Phi_T, PhiPPhi_T);
        sum_m (PhiPPhi_T,Q,Pp);

        delete_m(PhiP);
        delete_m(Phi_T);
        delete_m(PhiPPhi_T);
}

void ekf_u(struct wm *wm, struct ss_d *ss) {

        struct matrix *H, *P, *Pp, *R, *x, *xp, *z;
        struct matrix *h;
        struct matrix *deltaz;
        struct matrix *H_T, *PpH_T, *HPpH_T, *M, *invM, *K, *I, *KH, *IKH, *Kdeltaz;

        int nrz, ncz, nrH, ncH, nrR, ncR, nrPp, ncPp;

        H = ss->H;
        nrH = ss->H->n_row;
        ncH = ss->H->n_col;

        R = wm->R;
        nrR = wm->R->n_row;
        ncR = wm->R->n_col;

        P = wm->P;
        Pp = wm->Pp;
        nrPp = wm->Pp->n_row;
        ncPp = wm->Pp->n_col;

        x = ss->x;
        xp = ss->xp;

        z = ss->z;
        nrz = ss->z->n_row;
        ncz = ss->z->n_col;

        h =  matrix_m(nrz,ncz);
        deltaz = matrix_m(nrz,ncz);
        H_T = matrix_m(ncH, nrH);
        PpH_T = matrix_m(nrPp,nrH);
        HPpH_T = matrix_m(nrH, nrH);
        M = matrix_m(nrR,ncR);
        invM = matrix_m(nrR,ncR);
        K = matrix_m(nrPp, ncR);
        I = matrix_m(nrPp,ncPp);
        KH = matrix_m(nrPp,ncH);
        IKH = matrix_m(nrPp,ncH);
        Kdeltaz = matrix_m(nrPp,ncz);
        transp_m(H,H_T);
        times_m(Pp,H_T, PpH_T);
        times_m(H,PpH_T, HPpH_T);
        sum_m(HPpH_T, R, M);
        inv_m(M,invM);
        times_m(PpH_T,invM,K);
        eye_m(1,I);
        times_m(K,H,KH);
        less_m(I,KH,IKH);
        times_m(IKH,Pp,P);

        update_z(x->elements,h->elements);
        less_m(z,h,deltaz);

        times_m(K,deltaz,Kdeltaz);
        sum_m(xp,Kdeltaz,x);

        delete_m(h);
        delete_m(deltaz);
        delete_m(H_T);
        delete_m(PpH_T);
        delete_m(HPpH_T);
        delete_m(M);
        delete_m(invM);
        delete_m(K);
        delete_m(I);
        delete_m(KH);
        delete_m(IKH);
        delete_m(Kdeltaz);
}

void ekf_p_array(struct ss_d *ss) {
        // Declaration of the variables
        struct matrix *x, *xp, *Phi;
        int nrPhi, ncPhi;

        x  = ss->x;
        xp = ss->xp;
        Phi = ss->Phi;

        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;

        times_m(Phi,x,xp);
}

void ekf_u_array(struct wm *wm, struct ss_d *ss) {

        struct matrix *Phi, *G, *H, *P, *P_sr, *Q_sr, *R_sr, *x, *xp, *z;
        struct matrix *pre_array, *pos_array;
        struct matrix *h;
        struct matrix *deltaz;
        struct matrix *HPhi, *HPhiP_sr, *HG, *HGQ_sr, *PhiP_sr, *GQ_sr;
        struct matrix *ZEROS;

        int nrz, ncz, nrPhi, ncPhi, nrG, ncG, nrH, ncH;
        int nrQ_sr, ncQ_sr, nrR_sr, ncR_sr, nrP, ncP;

        int i,j;
        Phi = ss->Phi;
        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;

        G = ss->G;
        nrG = ss->G->n_row;
        ncG = ss->G->n_col;

        H = ss->H;
        nrH = ss->H->n_row;
        ncH = ss->H->n_col;

        Q_sr = wm->Qeta_sr;
        nrQ_sr = wm->Qeta_sr->n_row;
        ncQ_sr = wm->Qeta_sr->n_col;

        R_sr = wm->R_sr;
        nrR_sr = wm->R_sr->n_row;
        ncR_sr = wm->R_sr->n_col;

        P = wm->P;
        nrP = wm->P->n_row;
        ncP = wm->P->n_col;

        P_sr = wm->P_sr;


        x = ss->x;
        xp = ss->xp;

        z = ss->z;
        nrz = ss->z->n_row;
        ncz = ss->z->n_col;

        HPhi = matrix_m(nrH,ncPhi);
        HPhiP_sr = matrix_m(nrH,ncP);
        HG = matrix_m(nrH, ncG);
        HGQ_sr = matrix_m(nrH, ncQ_sr);
        ZEROS  = matrix_m(nrPhi,ncR_sr);
        PhiP_sr = matrix_m(nrPhi, ncP);
        GQ_sr = matrix_m(nrG, ncQ_sr);
        pre_array =  matrix_m((nrR_sr+nrPhi), (ncR_sr + ncP+ ncQ_sr));

        eye_m(0,ZEROS);
        equalpart_m(pre_array,0,0,R_sr);
        equalpart_m(pre_array,0,ncR_sr,HPhiP_sr);
        equalpart_m(pre_array,0,(ncR_sr+ncP),HGQ_sr);

        equalpart_m(pre_array,nrR_sr,0,ZEROS);
        equalpart_m(pre_array,nrR_sr,ncR_sr,PhiP_sr);
        equalpart_m(pre_array,nrR_sr,(ncR_sr+ncP),GQ_sr);

        givens_m(pre_array,pos_array);
        getpart_m(pos_array,ncR_sr,nrP,P_sr);

        times_m(P_sr,P_sr,P);

        delete_m(HPhi);
        delete_m(HPhiP_sr);
        delete_m(HG);
        delete_m(HGQ_sr);
        delete_m(ZEROS );
        delete_m(PhiP_sr);
        delete_m(GQ_sr);
        delete_m(pre_array);
        delete_m(pos_array);

}

