// Structs

struct matrix {
        double *elements;
        int n_row;
        int n_col;
};
//################################## IMU's structs #############################
struct axes {
        double x;
        double y;
        double z;
};
// Data from IMU
struct IMU{
        struct axes acel;
        struct axes rate;
        struct axes mag;
};

struct data_IMU {
        struct IMU bin;
        struct IMU scaled;
        struct IMU adj;
};

struct euler {
        double yaw;
        double pitch;
        double roll;
};

// ############################### State Space's structs ###############################
// Weight matrices
struct wm {
        struct matrix *Q;
        struct matrix *Qeta;
        struct matrix *Qeta_sr;
        struct matrix *R;
        struct matrix *R_sr;
        struct matrix *P;
        struct matrix *P_sr;
        struct matrix *Pp;
};
// Discrete space state
struct ss_d {
        struct matrix *x;
        struct matrix *xp;
        struct matrix *z;
        struct matrix *Phi;
        struct matrix *G;
        struct matrix *H;
        struct wm wm;
};

// #############################################################################
