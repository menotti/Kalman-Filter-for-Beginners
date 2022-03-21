// Prototypes of matrices.c

void updateOmega (struct matrix *omega, struct matrix *Omega);
void update_q(struct matrix *omega, struct matrix *q);
void update_rot(struct matrix *quat, struct matrix *Rot);
void update_z(double *q, double *z);
void sim_atitude (struct matrix *omega, struct matrix *q_sim, struct ss_d *sys_a);
void update_sys_a(struct matrix *omega_hat, struct ss_d *sys_a);
void updadate_wm_a(struct ss_d *ss_a, struct wm *wm_a);



