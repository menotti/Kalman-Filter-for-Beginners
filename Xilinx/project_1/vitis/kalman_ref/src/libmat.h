// Prototypes of libmat.c

struct matrix *matrix_m(int n_row, int n_col);
void delete_m(struct matrix *M);
void print_m(struct matrix *M);
void eye_m(double value, struct matrix *M);
short int equal_m(struct matrix *A, struct matrix *B);
void equalpart_m(struct matrix *A, int i_0, int j_0, struct matrix *B);
void getpart_m(struct matrix *A, int i_0, int j_0, struct matrix *B);
short int sum_m(struct matrix *A, struct matrix *B, struct matrix *result);
short int less_m(struct matrix *A, struct matrix *B, struct matrix *result);
short int times_m(struct matrix *A, struct matrix *B, struct matrix *result);
short int timesc_m(double value, struct matrix *A, struct matrix *result);
short int transp_m(struct matrix *A, struct matrix *result);
short int inv_m(struct matrix *A, struct matrix *result);
void givens_m(struct matrix *A, struct matrix *result);
void euler2quat_m(struct euler angles, struct matrix *quat);
