//---------------------------------------------------------------------------
#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"

//#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include "structs.h"
#include "defines.h"
#include "libmat.h"
#include "matrices.h"
#include "libkf.h"


int main()
{
          // Variables
    init_platform();

//    print("Hello Kalman\n\r");


        struct matrix *omega_g,*omega_hat, *bg, *m;
        struct matrix *q_sim, *rot_sim;
        struct euler angles;
        struct ss_d *sys_a = (struct ss_d*) malloc(sizeof(struct ss_d));
        struct wm *wm_a    = (struct wm*) malloc(sizeof(struct wm));

        int i, j, k, cont;




        omega_g   = matrix_m(3,1);
        omega_hat = matrix_m(3,1);
        bg      = matrix_m(3,1);
        m       = matrix_m(3,1);

        q_sim = matrix_m(4,1);
        rot_sim =  matrix_m(3,3);


        sys_a->x = matrix_m(7,1);
        sys_a->xp = matrix_m(7,1);
        sys_a->z = matrix_m(4,1);
        sys_a->Phi = matrix_m(7,7);
        sys_a->G = matrix_m(7,6);
        sys_a->H = matrix_m(4,7);

        wm_a->Q = matrix_m(7,7);
        wm_a->R = matrix_m(4,4);
        wm_a->P = matrix_m(7,7);
        wm_a->Pp = matrix_m(7,7);

        eye_m(1,wm_a->P);

        
        // #####################################################################
        // Initial conditions of the simulation

        omega_g->elements[0] = 0.1*angle2rad;
        omega_g->elements[1] = 0.3*angle2rad;
        omega_g->elements[2] = 0.1*angle2rad;

        angles.yaw   = 0;
        angles.pitch = 0;
        angles.roll  = PI/2 + PI/4;

        euler2quat_m(angles,q_sim);
        // #####################################################################
        for (cont = 0; cont <4; cont ++) {
                sys_a->x->elements[cont] = q_sim->elements[cont];
                bg->elements[cont] = 0;
        }
        for (cont = 0; cont <3; cont ++) {
                bg->elements[cont] = 0;
                sys_a->x->elements[cont + 4] = bg->elements[cont];
        }
        // #####################################################################
        for  (i = 0; i < 10000; i++) {
                sim_atitude (omega_g, q_sim, sys_a);
                less_m(omega_g, bg, omega_hat);

                update_sys_a(omega_hat, sys_a);
                updadate_wm_a(sys_a,wm_a);

                ekf_p(wm_a, sys_a);
                ekf_u(wm_a, sys_a);


                print_m(sys_a->x);
        }

        free(sys_a);

//        getchar();
//        getchar();
        cleanup_platform();
        return 0;
}
//---------------------------------------------------------------------------
