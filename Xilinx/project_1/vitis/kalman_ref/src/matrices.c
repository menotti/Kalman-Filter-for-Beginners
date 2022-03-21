// Matrices of the system
#include "stdlib.h"
#include "stdio.h"
#include "libmat.h"
#include "structs.h"
#include <math.h>
#include "defines.h"
#include "matrices.h"

void updateOmega (struct matrix *omega, struct matrix *Omega) {

        int ncO;
        double *pomega, *pOmega;

        ncO = Omega->n_col;

        pomega = omega->elements;
        pOmega = Omega->elements;

        // Omega line 0
        pOmega[0*ncO + 0] = 0;
        pOmega[0*ncO + 1] = -pomega[0];
        pOmega[0*ncO + 2] = -pomega[1];
        pOmega[0*ncO + 3] = -pomega[2];

        // Omega line 1
        pOmega[1*ncO + 0] =  pomega[0];
        pOmega[1*ncO + 1] =  0;
        pOmega[1*ncO + 2] =  pomega[2];
        pOmega[1*ncO + 3] = -pomega[1];

        // Omega line 2
        pOmega[2*ncO + 0] =  pomega[1];
        pOmega[2*ncO + 1] = -pomega[2];
        pOmega[2*ncO + 2] =  0;
        pOmega[2*ncO + 3] =  pomega[0];

        // Omega line 3
        pOmega[3*ncO + 0] =  pomega[2];
        pOmega[3*ncO + 1] =  pomega[1];
        pOmega[3*ncO + 2] = -pomega[0];
        pOmega[3*ncO + 3] =  0;
}

void update_q(struct matrix *omega, struct matrix *q){

        struct matrix *Omega, *COS, *SIN, *SUM, *q2;
        double *pomega, *pOmega, omega2;
        int nrO, ncO, nro, nco, nrq, ncq;
        int i;

        Omega = matrix_m(4,4);
        nrO = Omega->n_row;
        ncO = Omega->n_col;

        pomega = omega->elements;
        nro = omega->n_row;
        nco = omega->n_col;

        ncq = q->n_col;

        COS = matrix_m(nrO,ncO);
        SIN = matrix_m(nrO,ncO);
        SUM = matrix_m(nrO,ncO);
        q2  = matrix_m(nrO,ncq);

        omega2 =  pomega[0*nco + 0]*pomega[0*nco + 0];
        for (i=1; i<nro; i++) {
                omega2 =  omega2 + pomega[i*nco + 0]*pomega[i*nco + 0];
        }
        if (omega2  == 0) {
                printf("\n omega = 0 \n");
                omega2 = TINY;
        }
        updateOmega (omega, Omega);
        eye_m(cos(0.5*omega2*T),COS);
        timesc_m(1/omega2*sinf(0.5*omega2*T), Omega, SIN);
        sum_m(COS,SIN,SUM);
        times_m(SUM,q,q2);

        equal_m(q,q2);

        delete_m(Omega);
        delete_m(SIN);
        delete_m(COS);
        delete_m(SUM);
        delete_m(q2);
}


void update_rot(struct matrix *quat, struct matrix *Rot) {
        double *q, *pRot;
        int ncq, nrRot, ncRot;
        ncRot = Rot->n_col;

        q = quat->elements;
        pRot = Rot->elements;

        pRot[0*ncRot + 0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
        pRot[1*ncRot + 0] = 2*(q[1]*q[2] - q[0]*q[3]);
        pRot[2*ncRot + 0] = 2*(q[1]*q[3] + q[0]*q[2]);

        pRot[0*ncRot + 1] = 2*(q[1]*q[2] + q[0]*q[3]);
        pRot[1*ncRot + 1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
        pRot[2*ncRot + 1] = 2*(q[2]*q[3] + q[0]*q[1]);

        pRot[0*ncRot + 2] = 2*(q[1]*q[3] - q[0]*q[2]);
        pRot[1*ncRot + 2] = 2*(q[2]*q[3] + q[0]*q[1]);
        pRot[2*ncRot + 2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
}

void update_z(double *q, double *z) {
        z[0] = g*(2*(q[1]*q[3] - q[0]*q[2]));
        z[1] = g*(2*(q[2]*q[3] + q[0]*q[1]));
        z[2] = g*(q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]);
        z[3] = atan2((2*(q[1]*q[2] + q[0]*q[3])),(q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]));
}


void sim_atitude (struct matrix *omega, struct matrix *q_sim, struct ss_d *sys_a) {

        double *z, *x;

        z = sys_a->z->elements;
        x = sys_a->x->elements;

        update_q(omega, q_sim) ;
        update_z(x,z);
}

void update_sys_a(struct matrix *omega_hat, struct ss_d *sys_a)  {

        double *Phi, *G, *H, *omega, *q;
        double base, r00, r01;
        int ncPhi, ncG, ncH;

        Phi = sys_a->Phi->elements;
        ncPhi = sys_a->Phi->n_col;

        G = sys_a->G->elements;
        ncG = sys_a->G->n_col;

        H = sys_a->H->elements;
        ncH = sys_a->H->n_col;

        omega = omega_hat->elements;

        q = sys_a->x->elements;

        // Phi line 0
        Phi[0*ncPhi + 0] = 1 + 0;
        Phi[0*ncPhi + 1] = -0.5*T*omega[0];
        Phi[0*ncPhi + 2] = -0.5*T*omega[1];;
        Phi[0*ncPhi + 3] = -0.5*T*omega[2];;
        Phi[0*ncPhi + 4] = 0;
        Phi[0*ncPhi + 5] = 0;
        Phi[0*ncPhi + 6] = 0;


        // Phi line 1
        Phi[1*ncPhi + 0] =  0.5*T*omega[0];
        Phi[1*ncPhi + 1] =  1 + 0;
        Phi[1*ncPhi + 2] = -0.5*T*omega[2];
        Phi[1*ncPhi + 3] =  0.5*T*omega[1];
        Phi[1*ncPhi + 4] =  0;
        Phi[1*ncPhi + 5] =  0;
        Phi[1*ncPhi + 6] =  0;


        // Phi line 2
        Phi[2*ncPhi + 0] =  0.5*T*omega[1];
        Phi[2*ncPhi + 1] =  0.5*T*omega[2];
        Phi[2*ncPhi + 2] =  1 + 0;
        Phi[2*ncPhi + 3] = -0.5*T*omega[0];
        Phi[2*ncPhi + 4] =  0;
        Phi[2*ncPhi + 5] =  0;
        Phi[2*ncPhi + 6] =  0;


        // Phi line 3
        Phi[3*ncPhi + 0] =  0.5*T*omega[2];
        Phi[3*ncPhi + 1] = -0.5*T*omega[1];
        Phi[3*ncPhi + 2] =  0.5*T*omega[0];
        Phi[3*ncPhi + 3] =  1 + 0;
        Phi[3*ncPhi + 4] =  0;
        Phi[3*ncPhi + 5] =  0;
        Phi[3*ncPhi + 6] =  0;


        // Phi line 4
        Phi[4*ncPhi + 0] = 0;
        Phi[4*ncPhi + 1] = 0;
        Phi[4*ncPhi + 2] = 0;
        Phi[4*ncPhi + 3] = 0;
        Phi[4*ncPhi + 4] = 1 - T*1/tau;
        Phi[4*ncPhi + 5] = 0;
        Phi[4*ncPhi + 6] = 0;


        // Phi line 5
        Phi[5*ncPhi + 0] = 0;
        Phi[5*ncPhi + 1] = 0;
        Phi[5*ncPhi + 2] = 0;
        Phi[5*ncPhi + 3] = 0;
        Phi[5*ncPhi + 4] = 0;
        Phi[5*ncPhi + 5] = 1 - T*1/tau;
        Phi[5*ncPhi + 6] = 0;


        // Phi line 6
        Phi[6*ncPhi + 0] = 0;
        Phi[6*ncPhi + 1] = 0;
        Phi[6*ncPhi + 2] = 0;
        Phi[6*ncPhi + 3] = 0;
        Phi[6*ncPhi + 4] = 0;
        Phi[6*ncPhi + 5] = 0;
        Phi[6*ncPhi + 6] = 1 - T*1/tau;

        // G line 0
        G[0*ncG + 0] =  0.5*q[1];
        G[0*ncG + 1] =  0.5*q[2];
        G[0*ncG + 2] =  0.5*q[3];
        G[0*ncG + 3] =  0;
        G[0*ncG + 4] =  0;
        G[0*ncG + 5] =  0;

        // G line 1
        G[1*ncG + 0] = -0.5*q[0];
        G[1*ncG + 1] =  0.5*q[3];
        G[1*ncG + 2] = -0.5*q[2];
        G[1*ncG + 3] =  0;
        G[1*ncG + 4] =  0;
        G[1*ncG + 5] =  0;

        // G line 2
        G[2*ncG + 0] = -0.5*q[3];
        G[2*ncG + 1] = -0.5*q[0];
        G[2*ncG + 2] =  0.5*q[1];
        G[2*ncG + 3] =  0;
        G[2*ncG + 4] =  0;
        G[2*ncG + 5] =  0;

        // G line 3
        G[3*ncG + 0] =  0.5*q[2];
        G[3*ncG + 1] = -0.5*q[1];
        G[3*ncG + 2] = -0.5*q[0];
        G[3*ncG + 3] =  0;
        G[3*ncG + 4] =  0;
        G[3*ncG + 5] =  0;

        // G line 4
        G[4*ncG + 0] =  0;
        G[4*ncG + 1] =  0;
        G[4*ncG + 2] =  0;
        G[4*ncG + 3] =  1;
        G[4*ncG + 4] =  0;
        G[4*ncG + 5] =  0;

        // G line 5
        G[5*ncG + 0] =  0;
        G[5*ncG + 1] =  0;
        G[5*ncG + 2] =  0;
        G[5*ncG + 3] =  0;
        G[5*ncG + 4] =  1;
        G[5*ncG + 5] =  0;

        // G line 6
        G[6*ncG + 0] =  0;
        G[6*ncG + 1] =  0;
        G[6*ncG + 2] =  0;
        G[6*ncG + 3] =  0;
        G[6*ncG + 4] =  0;
        G[6*ncG + 5] =  1;

        H[ncH*0 + 0] = -2*g*q[2];
        H[ncH*0 + 1] =  2*g*q[3];
        H[ncH*0 + 2] = -2*g*q[0];
        H[ncH*0 + 3] =  2*g*q[1];
        H[ncH*0 + 4] =  0;
        H[ncH*0 + 5] =  0;
        H[ncH*0 + 6] =  0;

        H[ncH*1 + 0] =  2*g*q[1];
        H[ncH*1 + 1] =  2*g*q[0];
        H[ncH*1 + 2] =  2*g*q[3];
        H[ncH*1 + 3] =  2*g*q[2];
        H[ncH*1 + 4] =  0;
        H[ncH*1 + 5] =  0;
        H[ncH*1 + 6] =  0;

        H[ncH*2 + 0] =  2*g*q[0];
        H[ncH*2 + 1] = -2*g*q[1];
        H[ncH*2 + 2] = -2*g*q[2];
        H[ncH*2 + 3] =  2*g*q[3];
        H[ncH*2 + 4] =  0;
        H[ncH*2 + 5] =  0;
        H[ncH*2 + 6] =  0;

        r00 = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
        r01 = 2*(q[1]*q[2] + q[0]*q[3]);

        base =  r00*r00 + r01*r01;
        H[ncH*3 + 0] = (2*q[3]*r00 - 2*q[0]*r01)/base;
        H[ncH*3 + 1] = (2*q[2]*r00 - 2*q[1]*r01)/base;
        H[ncH*3 + 2] = (2*q[1]*r00 + 2*q[2]*r01)/base;
        H[ncH*3 + 3] = (2*q[0]*r00 + 2*q[3+ 2]*r01)/base;
        H[ncH*3 + 4] =  0;
        H[ncH*3 + 5] =  0;
        H[ncH*3 + 6] =  0;

}

void updadate_wm_a(struct ss_d *ss_a, struct wm *wm_a){

        struct matrix *Qeta, *PhiG, *Phi_T, *G_T, *G_TPhi_T, *auxQ;
        struct matrix *Phi, *G, *Q, *R;
        int nrPhi, ncPhi, nrG, ncG, ncQ, ncR, ncQeta;

        Phi = ss_a->Phi;
        nrPhi = ss_a->Phi->n_row;
        ncPhi = ss_a->Phi->n_col;

        G = ss_a->G;
        nrG = ss_a->G->n_row;
        ncG = ss_a->G->n_col;

        Q = wm_a->Q;
        R = wm_a->R;
        ncR = wm_a->R->n_col;

        Qeta = matrix_m(6,6);
        ncQeta = Qeta->n_col;

        eye_m(0, Qeta);  // It fills out the matrix with zeros;

        Qeta->elements[ncQeta*0 + 0] = 1.6106;
        Qeta->elements[ncQeta*1 + 1] = 1.6748;
        Qeta->elements[ncQeta*2 + 2] = 0.2819;
        Qeta->elements[ncQeta*3 + 3] = 111.6747;
        Qeta->elements[ncQeta*4 + 4] = 149.4054;
        Qeta->elements[ncQeta*5 + 5] = 150.3087;

        PhiG = matrix_m(nrPhi,ncG);
        Phi_T = matrix_m(ncPhi, nrPhi);
        G_T = matrix_m(ncG,nrG);
        G_TPhi_T = matrix_m(ncG,nrPhi);
        auxQ = matrix_m(nrPhi,ncG);

        times_m(Phi, G, PhiG);
        transp_m(Phi,Phi_T);
        transp_m(G,G_T);
        times_m(G_T,Phi_T,G_TPhi_T);
        times_m(PhiG, Qeta, auxQ);
        times_m(auxQ,G_TPhi_T,Q);

        eye_m(0,R);    // It fills out the matrix with zeros;
        R->elements[ncR*0 + 0] = 1.6106;
        R->elements[ncR*1 + 1] = 1.6106;
        R->elements[ncR*2 + 2] = 1.6106;
        R->elements[ncR*3 + 3] = 1.6106;

        delete_m(Qeta);
        delete_m(PhiG);
        delete_m(Phi_T);
        delete_m(G_T);
        free(G_TPhi_T);
        free(auxQ);
}






