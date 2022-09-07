#include "libkf.hpp"


// ============ Fx and Hx Definitions ============ 
// template<typename T = float>
void fx(Matrix<_Tp> *Xi, double dt, Matrix<_Tp> *fXi, int itr, int x_stt)
{

        // Matrix<_Tp> *xp;
        // Matrix<_Tp> *xt;
        // A = [1 dt  0] [a]   [a + dt x b]
        //     [0  1  0] [b] = [b         ]
        //     [0  0  1] [c]   [c         ]

        for(int i = 0; i < x_stt; i++)
                fXi->at(itr, i) = Xi->at(itr, i);
        fXi->at(itr, 0) += dt * Xi->at(itr, 1);        

}

// template<typename T = float>
void hx(Matrix<_Tp> *Xi, Matrix<_Tp> *hXi, int itr)
{
        _Tp x0 = Xi->at(itr, 0);
        _Tp x2 = Xi->at(itr, 2);

        hXi->at(itr, 0) = sqrt(x0*x0 + x2*x2);
}
// ================================================ 

// template<typename T = float> 
ss_d *init_ssd(const int x_stt, const int z_stt, _Tp dt)
{
    ss_d* ss = new struct ss_d;
    ss->x_states = x_stt;
    ss->z_states = z_stt;
    int kmax = 2*x_stt+1;

    ss->dt = dt;
    

    //allocate only once time
    ss->Xi  = new Matrix<_Tp>(kmax, x_stt);
    ss->fXi = new Matrix<_Tp>(kmax, x_stt);
    ss->hXi = new Matrix<_Tp>(kmax, z_stt);
    
    ss->z = new Matrix<_Tp>(z_stt, 1);
    ss->x = new Matrix<_Tp>(x_stt, 1);
    ss->zp = new Matrix<_Tp>(z_stt, 1);
    ss->xp = new Matrix<_Tp>(x_stt, 1);
    
    ss->K = new Matrix<_Tp>(x_stt, z_stt);
        
    return ss;
}


// template <class T = float>
wm *init_wm(const int x_stt, const int z_stt, _Tp kappa)
{
    //create wm pointer
    struct wm *wm = new struct wm;
    //declare matrices
    wm->Q = new Matrix<_Tp>(x_stt, x_stt); 
    wm->R = new Matrix<_Tp>(z_stt, z_stt);
    wm->P = new Matrix<_Tp>(x_stt, x_stt);
    wm->Pp = new Matrix<_Tp>(x_stt, x_stt);
    wm->Pz = new Matrix<_Tp>(z_stt, z_stt);
    wm->Pxz = new Matrix<_Tp>(x_stt, z_stt);


    wm->W = new std::vector<_Tp>(2*x_stt+1);
    wm->kappa = kappa;

    return wm;
}

// template<typename Tp>
void UT(Matrix<_Tp> *Xi, std::vector<_Tp> *W, Matrix<_Tp> *NoiseCov,
        Matrix<_Tp> *xm, Matrix<_Tp> *xcov, int n){

        

        int kmax = W->size();
        // kmax = std::fmax(Xi->n_row, Xi->n_col);        

        //matrix pointers - auxiliary matrices to delete afterwards.. 
        Matrix<_Tp> *aux = new Matrix<_Tp>(n, 1);
        Matrix<_Tp> *aux_t = new Matrix<_Tp>(1, n);
        Matrix<_Tp> *aux_xcov = new Matrix<_Tp>(n, n);
        
        // xm  = new Matrix<_Tp>(n, 1); //if non-zero matrix...
        // xcov = new Matrix<_Tp>(n, n);
        
        xm->zeros();
        //Xm accumulation
        for(int i = 0; i < kmax; i++){
                for(int j = 0; j < n; j++){
                        xm->at(j, 0) += W->at(i) * Xi->at(i, j);
                        // std::cout << W->at(i) << "*" << Xi->at(i, j) << "\n";
                        //xm and W being treated as vector..
                }
        }
        // std::cout <<" xm ";
        // xm->display();

        //Xcov Accumulation
        // xcov = xcov + W(k)*(Xi(:, k) - xm)*(Xi(:, k) - xm)_t;
        // std::cout <<"xm "; 
        // xm->display();
        // std::cout <<"Xi "; 
        // Xi->display();
        for(int i = 0; i< kmax; i++){
                
                for(int j = 0; j != n; j++)
                        aux->elem[j] = (Xi->at(i, j) - xm->elem[j]);
                
                // std::cout <<"\n";
                // std::cout <<"aux - "<< aux->n_row <<" x " << aux->n_col;
                // aux->display();
                transp_m(aux, aux_t);
                times_m(aux, aux_t, aux_xcov);

                timesc_m(W->at(i), aux_xcov, aux_xcov);

                sum_m(xcov, aux_xcov, xcov);
        }
        // std::cout <<"last aux_xcov";
        // aux_xcov->display();
        // std::cout <<"last aux_t";
        // aux_t->display();
        sum_m(xcov, NoiseCov, xcov);
        // std::cout <<"xcov - "; 
        // xcov->display();

        delete aux;
        delete aux_t;
        delete aux_xcov;
}


// template <typename T = float>
void set_weights(std::vector<_Tp> *W, _Tp kappa, int stt)
{
        int kmax = 2*stt+1;
        W->at(0) = kappa / (stt+ kappa); 
        for(int i = 1; i < kmax; i++)
                W->at(i) = 1/(2*(stt+kappa));
}

// template<typename T = float>
void sigma_pts(ss_d *ss, wm *wm){
        //#sigmaPts[0] = x
        //#sigmaPts[i] = x + U[i]  , with i = 1,2, ..., (M-1)
        //#sigmaPts[i] = x - U[i]  , with i = M, M+1, ..., 2M. 
        
        int n = ss->x_states;
        
        int kmax = 2*n+1;
        _Tp kappa = wm->kappa;
        Matrix<_Tp> *U = new Matrix<_Tp>(n,n);

        _Tp scalar = n+kappa;
        cholesky(wm->P, U, scalar);
        

        //copy all x values in Xi
        for(int i = 0; i < kmax; i++){
                for(int k = 0; k < n; k++ )
                        ss->Xi->at(i,k) = ss->x->elem[k];
        }
        
        //makes diff (and sum) between Xi and U
        for(int i = 0; i < n; i++){
                for(int k = 0; k < n; k++ ){
                        ss->Xi->at(  i+1, k) += U->at(i, k);
                        ss->Xi->at(n+i+1, k) -= U->at(i, k);
                        
                }
        }
        // std::cout <<" Xi Matrix\n ";
        // ss->Xi->display();   
}

void ukf_predict(struct wm *wm, struct ss_d *ss)
{
        _Tp dt = ss->dt;
        _Tp kappa = wm->kappa;
        // double kappa = wm->kappa;
        int n = ss->x_states;
        int nz = ss->z_states;
        
        int kmax = 2*n+1;

        //I.Compute Sigma Weights 
        set_weights(wm->W, kappa, n);
        sigma_pts(ss, wm);
        
        //todo: parallelizable loop...
        for(int i = 0; i < kmax; i++){
                fx(ss->Xi, dt, ss->fXi, i, n);
                hx(ss->Xi, ss->hXi, i);
        }       
        // ss->fXi->display();
        // ss->hXi->display();
        //todo: parallelize steps II and III depending on step I
        //II. Predict state & error covariance
        UT(ss->fXi, wm->W, wm->Q , ss->xp, wm->Pp, n);
        // std::cout <<" xp e Pp\n"; 
        // ss->xp->display();
        // wm->Pp->display();
        //III. Predict measurement & covariance
        // std::cout<< "display input UT2\n";
        // wm->R->display();
        // ss->hXi->display(); 
        
        UT(ss->hXi, wm->W, wm->R , ss->zp, wm->Pz, nz);
        // std::cout <<" zp e Pz\n"; 
        // ss->zp->display();
        // wm->Pz->display();
        //End of Prediction
}

// template<typename Tp>
void ukf_update(struct wm *wm, struct ss_d *ss)
{
        

        int n = ss->x_states; 
        int m = ss->z_states;
        
        // int n = std::fmax(ss->x->n_col, ss->x->n_row); //todo:better way to define it?
        // int m = std::fmin(ss->z->n_col, ss->z->n_row); //todo:better way to define it?
        
        Matrix<_Tp> *a = new Matrix<_Tp>(n, 1); 
        Matrix<_Tp> *b = new Matrix<_Tp>(1, m);

    
        Matrix<_Tp> *Pz_i = new Matrix<_Tp>(wm->Pz->n_row, wm->Pz->n_col); 
        Matrix<_Tp> *aux_K = new Matrix<_Tp>(n,wm->Pz->n_col);
        Matrix<_Tp> *aux_Pxz = new Matrix<_Tp>(n, b->n_col);
        Matrix<_Tp> *zzp  =  new Matrix<_Tp>(ss->z->n_row, ss->z->n_col);

        
        // Pxz += W(i) *{f(Xi) - xp} x {h(Xi)-zp}_t
        wm->Pxz->zeros();
        for (int i = 0; i < 2*n+1; i++){
//             print_m(*ss->zp);
                for (int k = 0; k < n; k++){
                        a->elem[k] = ss->fXi->at(i, k) - ss->xp->elem[k];
                }
                
                for (int k = 0; k < m; k++){
                        b->elem[k] = ss->hXi->at(i, k) - ss->zp->elem[k];
                }
//                 print_m(*a);
//                 print_m(*b);
//             std::cout << "----\n";
                times_m(a,b,aux_Pxz);
                timesc_m(wm->W->at(i), aux_Pxz, aux_Pxz);
                sum_m(aux_Pxz, wm->Pxz, wm->Pxz);
        }       

        // std::cout <<" Pxz Matrix\n";
        // wm->Pxz->display();
//         print_m(*wm->Pz);
        inv_m(wm->Pz, Pz_i);
        // wm->Pz->display();
        // std::cout <<" Px_i\n";
        // Pz_i->display();
        
        ss->K->zeros();
        //K = Pxz * inv(Pz)
        times_m(wm->Pxz, Pz_i, ss->K);
        // ss->K->display();
        
        // x = xp + K * (z-zp)
        less_m(ss->z, ss->zp, zzp);
        times_m(ss->K, zzp, aux_K);
        
        sum_m(ss->xp, aux_K, ss->x);

        delete (zzp);
        delete (aux_K);
        delete (Pz_i);
}