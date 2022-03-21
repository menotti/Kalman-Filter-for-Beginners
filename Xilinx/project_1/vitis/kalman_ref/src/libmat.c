// Libray of matrices

#include "structs.h"
#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include "libmat.h"
#include "defines.h"

// Create a matrix
struct matrix *matrix_m(int n_row, int n_col) {
        struct matrix *M = (struct matrix *) malloc(sizeof(M));
        M->elements = (double*) malloc((n_row*n_col)*sizeof(double));
        M->n_row = n_row;
        M->n_col = n_col;
        return M;
}
// Delete a matrix
void delete_m(struct matrix *M) {
        free(M->elements);
        free(M);
}
// Print a matrix
void print_m(struct matrix *M) {
        double *p;
        int nr, nc;
        int i,j;

        p =  M->elements;
        nr = M->n_row;
        nc = M->n_col;

        for (i=0; i<nr; i++)  {
                for (j=0; j<nc; j++)  {
                        if (j == nc -1) {
                                printf("%0.8f ", p[i*nc + j]);
                                printf(" \n " );
                        }
                        else {
                                printf("%0.8f ", p[i*nc + j]);
                        }
                }
        }
        printf("\n");
}

// Eye of a matrix
void eye_m(double value, struct matrix *M) {

        double *p;
        int nr, nc;
        int i,j;

        p = M->elements;
        nr = M->n_row;
        nc = M->n_col;

        for (i=0; i<nr; i++) {
                for (j=0; j<nc; j++) {
                        if (i == j) {
                                p[i*nc + j]=value;
                        }
                        else {
                                p[i*nc + j]=0;
                        }
                }
        }
}

// Equality of matrix
short int equal_m(struct matrix *A, struct matrix *B) {
        int nrA, ncA, nrB, ncB;
        int i,j;
        double *pA, *pB;

        pA = A->elements;
        pB = B->elements;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;

        if (nrA != nrB | ncA != ncB) {
                printf("\n Wrong dimension \n") ;
                return 0;
        }
        else {
                for (i = 0; i< nrA; i++) {
                        for (j = 0; j< ncA; j++) {
                                pA[i*ncA + j] = pB [i*ncB + j];
                        }
                }
                return 1;
        }
}

void equalpart_m(struct matrix *A, int i_0, int j_0, struct matrix *B) {
        int nrA, ncA, nrB, ncB;
        int nr_f, nc_f;
        int i_A,j_A, i_B, j_B;
        double *pA, *pB;



        pA = A->elements;
        pB = B->elements;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;

        nr_f = (i_0 + nrB);  // final row
        nc_f = (j_0 + ncB);  // final column

        for (i_A = i_0, i_B =0; i_A < nr_f; i_A++, i_B++) {
                for (j_A = j_0, j_B = 0; j_A < nc_f; j_A++, j_B++) {
                        pA[i_A*ncA + j_A] = pB [i_B*ncB + j_B];
                }
        }

}

void getpart_m(struct matrix *A, int i_0, int j_0, struct matrix *B) {

        int nrA, ncA, nrB, ncB;
        int nr_f, nc_f;
        int i_A,j_A, i_B, j_B;
        double *pA, *pB;

        pA = A->elements;
        pB = B->elements;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;

        nr_f = (i_0 + nrB);  // final row
        nc_f = (j_0 + ncB);  // final column

        for (i_A = i_0, i_B =0; i_A < nr_f; i_A++, i_B++) {
                for (j_A = j_0, j_B = 0; j_A < nc_f; j_A++, j_B++) {
                         pB [i_B*ncB + j_B] = pA[i_A*ncA + j_A];
                }
        }

}


// Sum of matrices
short int sum_m(struct matrix *A, struct matrix *B, struct matrix *result) {

        double *pA, *pB, *pR;
        int nrA, nrB, nrresult, ncA, ncB, ncresult;
        int cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pB = B->elements;
        pR = result->elements;

        if (nrA != nrB | ncA != ncB | nrresult != nrA | ncresult != ncA) {
                printf("Wrong dimension");
                return 0;
        }
        else {
                for(cont = 0; cont<(nrA*ncA); cont++) {
                        pR[cont] = pA[cont] + pB[cont];
                }
                return 1;
        }

}

// Less of matrix
short int less_m(struct matrix *A, struct matrix *B, struct matrix *result) {

        double *pA, *pB, *pR;
        int nrA, nrB, nrresult, ncA, ncB, ncresult;
        int cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pB = B->elements;
        pR = result->elements;

        if (nrA != nrB | ncA != ncB | nrresult != nrA | ncresult != ncA) {
                printf("Wrong dimension");
                return 0;
        }
        else {
                for(cont = 0; cont<(nrA*ncA); cont++) {
                        pR[cont] = pA[cont] - pB[cont];
                }
                return 1;
        }

}

// Times between two matrices
short int times_m(struct matrix *A, struct matrix *B, struct matrix *result) {

        double *pA, *pB, *pR, aux;
        int nrA, nrB, nrresult, ncA, ncB, ncresult;
        int i, j, cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrB = B->n_row;
        ncB = B->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pB = B->elements;
        pR = result->elements;

        if (ncA != nrB |  nrresult != nrA | ncresult != ncB) {
                printf("Wrong dimension");
                return 0;
        }
        else {
                for (i = 0; i<nrA; i++) {
                        for (j = 0; j<ncB; j++) {
                                aux = 0;
                                for (cont=0; cont<nrB; cont++ ) {
                                    aux=aux + pA[ncA*i + cont]*pB[ncB*cont + j];
                                }
                                pR[ncB*i + j]= aux;
                        }
                }
                return 1;
        }
}

// Times between a constant and a matrix
short int timesc_m(double value, struct matrix *A, struct matrix *result) {

        double *pA, *pB, *pR, aux;
        int nrA, nrresult, ncA, ncresult;
        int cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pR = result->elements;

        if (nrresult != nrA | ncresult != ncA) {
                printf("Wrong dimension");
                return 0;
        }
        else {
                for (cont=0; cont<(nrA*ncA); cont++ ) {
                        pR[cont] = value*pA[cont];
                }
                return 1;
        }
}
// Transpose of a matrix
short int transp_m(struct matrix *A, struct matrix *result) {

        double *pA, *pB, *pR;
        int nrA, nrresult, ncA, ncresult;
        int i, j, cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pR = result->elements;

        if (nrresult != ncA | ncresult != nrA) {
                printf("Wrong dimension");
                return 0;
        }
        else {
                for (i=0; i<nrA; i++) {
                        for (j=0; j<ncA; j++) {
                                pR[nrA*j+i] = pA[ncA*i+j];
                        }
                }
                return 1;
        }
}


// Inverse of a matrix
short int inv_m(struct matrix *A, struct matrix *result) {

        double *pA, *pB, *pR, *pa;
        double sum, aux, *b, *x;
        struct matrix *a;
        int nrA, nrresult, ncA, ncresult;
        int idx2, *idx, mem, flag;
        int i,j,k, cont;
        nrA = A->n_row;
        ncA = A->n_col;
        nrresult = result->n_row;
        ncresult = result->n_col;
        pA = A->elements;
        pR = result->elements;

        a = matrix_m(nrA,ncA);
        pa = a->elements;

       if (nrresult != nrA | ncresult != ncA) {
                printf("Wrong dimension");
                return 0;
        }
        else {

        for (i = 0; i<nrA; i++) {
                for (j = 0; j<nrA; j++) {
                        pa[ncA*i+j] = pA[ncA*i+j];
                }
        }
//---------------------------- Partial pivoting --------------------------------
        b = (double*) malloc(nrA*sizeof(double));
        x = (double*) malloc(nrA*sizeof(double));
        idx = (int*) malloc(nrA*sizeof(int));
        for (k = 0; k<nrA; k++)
                idx[k] = k;

        for (i = 0; i<nrA; i++) {
                j = i;
                idx2 = i;
                if (pa[ncA*i+j] == 0) {
                        flag = 1;
                        for (k = i+1; k<nrA; k++ ) {
                                if (fabs(pa[ncA*k+j]) >= TINY && flag == 1) {
                                        mem  = idx[i];
                                        idx[i] = idx[k];
                                        idx[k] = mem;
                                        idx2 = k;
                                        flag = 0;
                                }
                        }
                        if (flag == 1) {
                                for (k = 0; k<nrA; k++) {
                                        if (fabs(pa[ncA*k+j]) > TINY && fabs(pa[ncA*i+k]) > TINY) {
                                                mem = idx[i];
                                                idx[i] = idx[k];
                                                idx[k] = mem;
                                                idx2 = k;
                                                flag = 0;
                                        }
                                }
                        }
                        if (idx2 == i){
                                printf("\n Singular matrix \n \n");
                                pa[ncA*i+j] = TINY;
                        }
                        for (k = 0; k<nrA; k++){
                                mem = pa[ncA*i+k];
                                pa[ncA*i+k] = pa[ncA*idx2+k];
                                pa[ncA*idx2+k] = mem;
                        }
                }

        }

//------------------- Crout's algorithm for LU Decomposition -------------------
        for (j = 0; j<nrA; j++) {
                for (i = 0; i<nrA; i++) {
                        if (i<j | i ==j) {
                                sum = pa[ncA*i+j];
                                for (k = 0; k<i; k++) {
                                        sum = sum - pa[ncA*i+k]*pa[ncA*k+j];
                                }
                                pa[ncA*i+j] = sum;
                        }
                        if (i > j) {
                                sum = pa[ncA*i+j];
                                for (k = 0; k<j; k++) {
                                        sum = sum - pa[ncA*i+k]*pa[ncA*k+j];
                                }
                                pa[ncA*i+j] = sum/pa[ncA*j+j];
                        }
                }
        }
//---------------------------- Forward substituion -----------------------------
        for (k = 0; k<nrA; k++) {
                for (cont = 0; cont<nrA; cont ++ ) {
                        b[cont] = 0;
                }
                b[k] = 1;
                for (i = 0; i<nrA; i++) {
                        sum = b[i];
                        for (j = 0; j<i; j++) {
                                sum = sum - pa[ncA*i+j]*x[j];
                        }
                        x[i] = sum;
                }
//---------------------------- Backward substituion ----------------------------
                for (i=(nrA-1); i>=0; i--) {
                        sum = x[i];
                        for (j = i+1; j<nrA; j++) {
                                sum = sum - pa[ncA*i+j]*x[j];
                        }
                        x[i] = sum/pa[ncA*i+i];
                }
                for (cont = 0;  cont<nrA; cont++){
                        pR[ncA*cont+idx[k]] = x[cont];
                }
        }
        delete_m(a);
        free(b);
        free(x);
        free(idx);
        return 1;
        }
}

// Givens of a matrix
void givens_m(struct matrix *A, struct matrix *result) {

        struct matrix *Theta;
        int nrA, ncA, i, j, i2, j2, cont;
        double a, b, c, rho;
        double *pA, *pTheta;
        short int flag;

        nrA = A->n_row;
        ncA = A->n_col;
        pA = A->elements;

        Theta = matrix_m(ncA,ncA);
        pTheta = Theta->elements;

        for (i =0; i<nrA; i++){
                for (j = ncA-1; j>=i+1; j--) {
                        b = pA[ncA*i+j];
                        flag = 0;
                        for (cont = i; cont < j; cont ++) {
                                a = pA[ncA*i+cont];
                                if (fabs(a) >= TINY) {
                                        flag = 1;
                                        break ;
                                }
                        }
                        if (flag ==0) {
                                a = TINY;
                                printf("\n a = 0 \n");
                        }
                        rho = b/a;
                        for (i2 =0; i2<ncA; i2++){
                                for (j2 =0; j2<ncA; j2++){
                                        if (i2 == j2) {
                                                pTheta[ncA*i2+j2] = 1;
                                        }
                                        else {
                                                pTheta[ncA*i2+j2] = 0;
                                        }
                                }
                        }
                        c = 1/sqrt(1 +rho*rho);
                        pTheta[ncA*cont+cont]  = c;
                        pTheta[ncA*cont+j]  = -rho*c;
                        pTheta[ncA*j+cont]  = rho*c;
                        pTheta[ncA*j+j]  = c;

                        times_m(A, Theta, result);
                }
        }
}

void euler2quat_m(struct euler angles, struct matrix *quat) {

        double *q, psi, theta, phi;
        int nrq, ncq;

        psi   =  angles.yaw;
        theta =  angles.pitch;
        phi   =  angles.roll;

        q = quat->elements;

        q[0] = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2);
        q[1] = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2);
        q[2] = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2);
        q[3] = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2);
}
