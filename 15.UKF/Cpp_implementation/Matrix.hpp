#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream> 
#include <cstring>
#include <cmath> /*using std::abs */ 
#include "Defines.h"

template< class Tp>
class Matrix
{
    // private:
    // protected:
    public:
    std::vector<Tp> elem;
    int n_row, n_col;
    
    public:
    // int col() const {{return n_col;}}
    // int row() const {{return n_row;}}
    // Returns size matrix

    // Constructor
    Matrix(int row, int col)
    {
        n_col = col;
        n_row = row;
        elem.resize(n_col * n_row, 0.0);
    }

    // Destructor
    ~Matrix()
    {
        elem.clear();
    }

    friend bool operator!=(const Matrix &A, const Matrix &B)
    {
        return ((A.n_row != B.n_row || A.n_col != B.n_row) || A.elem != B.elem);
    }
    bool is_symmetric();

    bool is_square();

    //memset to fill up
    void zeros();

    void eye(Tp alpha);
    
    void copy_of(Matrix &A);

    void getT();

    int size(){return this->elem.size();};

    bool equal_dim(Matrix &B){
        return ((n_row == B.n_row) && (n_col == B.n_col));}
    
    //Pointers & address intructions...
    Tp &at(int i, int j){return this->elem[n_col*i+j];}
    Tp *data() {return elem.data();}


    void display();
};

// template<typename Tp>
// void transp_m(Matrix<Tp> *A, Matrix<Tp> *B);

// template <typename Tp>
// bool timesc_m(double val, Matrix<Tp> *A, Matrix<Tp> *R);

// template<typename Tp>
// void times_m(Matrix<Tp> *A, Matrix<Tp> *B, Matrix<Tp> *C);

// template <typename Tp>
// bool cholesky(Matrix<Tp> *A, Matrix<Tp> *L, Tp alpha);

// template<typename Tp>
// bool sum_m(Matrix<Tp> *A, Matrix<Tp> *B, Matrix<Tp> *C);

// template<typename Tp>
// bool less_m(Matrix<Tp> *A, Matrix<Tp> *B, Matrix<Tp> *C);

//Check Symmetry
template<typename Tp>
bool Matrix<Tp>::is_symmetric()
{
    for(int i = 0; i < n_row; i++)
    {
        for(int j = 0; i< n_col; j++)
        {
            if(elem[n_row*i+j] != elem[n_row*j+i]) return false;
        }
    }
    return true;

}

//Check if is squared
template<typename _Tp>
bool Matrix<_Tp>::is_square()
    {
        return n_row == n_col;
    }


//Zero Initialization
template<typename _Tp>
void Matrix<_Tp>::zeros(){
        std::memset(elem.data(), _Tp{0}, sizeof(elem));
    }

//Eye Matrix.. 
template<typename Tp>
void Matrix<Tp>::eye(Tp alpha)
{   
    for (int j = 0; j < n_col; j++){
        for(int i=0; i < this->n_row; i++)
            if(i == j) this->at(i,j) = alpha;
            else this->at(i,j) = 0.0;
    }
}



//Creates a copy of matrix A
template<typename _Tp>
void Matrix<_Tp>::copy_of(Matrix<_Tp> &A)
{
  n_row = A.n_row;
  n_col = A.n_col;
  this->elem.clear();
  this->elem = A.elem;
}

template<typename _Tp>
void Matrix<_Tp>::getT()
{
  std::vector<_Tp> _aux(this->size());
  if(n_col != 1 || n_row != 1)
  {
    for(int i = 0; i < n_row; i++)
    {
      for(int j=0; j < n_col; j++)
      {
        _aux[n_col*i+j] = this->elem[n_col*j+i];
      }
    }
    this->elem.clear();
    this->elem = _aux;
  }
  // else n_col == 1 or n_row == 1, nothing changes in vec..
  int _row = n_row;
  n_row = n_col;
  n_col = _row;  
}

template<typename _Tp>
void Matrix<_Tp>::display()
{

    std::cout <<"Display Matrix\n";
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
        std::cout<< this->at(i, j) << " ";
        }
        std::cout<< "\n";
    }
    std::cout<< "\n";
}

template<typename _Tp>
void times_m(Matrix<_Tp> *A, Matrix<_Tp> *B, Matrix<_Tp> *C)
{  
    _Tp sum;
    int i,j,k;
    for (i = 0; i < A->n_row; i++)
    {
        for (j = 0; j < B->n_col; j++)
        {
            sum = 0.0;
            for (k = 0; k < B->n_row; k++)
            {
                sum += A->at(i, k) * B->at(k, j);
            }
            C->at(i, j) = sum;
        }
    }

}

//todo: Matrix X Vector multiplication that returns vector ?

template<typename _Tp>
bool transp_m(Matrix<_Tp> *A, Matrix<_Tp> *R)
{
    if (A->n_row != R->n_col || A->n_col != R->n_row)
    {
        std::cout << "Not transposed!\n";
        return false;
    }

    for (int i = 0; i < A->n_row; i++)
    {
        for (int j = 0; j < A->n_col; j++)
        {
            // C[i,j] = A[j,i]
            R->at(j, i) = A->at(i, j);
        }
    }
    return 1;
}

template<typename Tp> 
bool timesc_m(Tp val, Matrix<Tp> *A, Matrix<Tp> *R)
{
    if (A->size() != R->size())
        return false;

    for (int i = 0; i < A->size(); i++)
    {
        R->elem[i] = val * A->elem[i];
    }
    return true;
}


template <typename Tp>
bool cholesky(Matrix<Tp> *A, Matrix<Tp> *L, Tp alpha )
{

    if (!A.is_symmetric())
    {
       std::cout << "\nMatrix not symmetric\n";
       return 0;
    }
    Tp sum;
    int i, j, k;
    auto n = A->n_row;
    L->zeros();

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
        {
            sum = 0.0;
            if (i == j)//diagonals
            {
                for(k= 0; k < j; k++)
                    sum+= pow(L->elem[n*j+k], 2);
                L->elem[n*j+j] =  sqrt(alpha * A->elem[n*j+j]-sum);
                
            } else {
                //Evaluate L(i,j) using L(j, j)
                for(k = 0; k < j; k++)
                    sum+= L->elem[n*i+k] * L->elem[n*j+k];
                L->elem[n*i+j] = (alpha * A->elem[n*i+j] - sum)/L->elem[n*j+j];

            }
                
        }
    }
    
    return true;
}

template<typename Tp>
bool sum_m(Matrix<Tp> *A, Matrix<Tp> *B, Matrix<Tp> *C)
{
    // check sizes
    if (A->size() != B->size() || A->size() != C->size())
        return false;
    
    
    for (int i = 0; i < A->size(); i++)
    {
        C->elem[i] = B->elem[i] + A->elem[i];
    }
    return true;
}

template<typename Tp>
bool less_m(Matrix<Tp> *A, Matrix<Tp> *B, Matrix<Tp> *C)
{
    // check sizes
    if (A->size() != B->size() || A->size() != C->size())
        return false;
    
    
    for (int i = 0; i < A->size(); i++)
    {
        C->elem[i] = B->elem[i] - A->elem[i];
    }
    return true;
}

template<typename Tp>
short int inv_m(Matrix<Tp> *A, Matrix<Tp> *R)
{
    if (A->n_row != R->n_row || A->n_col != R->n_col)
    {
        std::cout << "Different dimension !\n ";
        return false;
    }

    //---------------------------- Partial pivoting --------------------------------
    int i, j, k, cont;
    int idx2, mem, flag;
    Tp sum;
    std::vector<Tp> b(A->n_row);
    std::vector<Tp> x(A->n_row);
    std::vector<int> idx(A->n_row);

    // Copy matrix<_Tp> - auxiliar
    Matrix<Tp> *a = new Matrix<Tp>(A->n_row, A->n_col);
    a->elem = A->elem;

    for (k = 0; k < A->n_row; k++)
        idx[k] = k;

    for (i = 0; i < A->n_row; i++)
    {
        j = i;
        idx2 = i;
        if (a->elem[A->n_col * i + j] == 0)
        {
            flag = 1;
            for (k = i + 1; k < A->n_row; k++)
            {
                if (std::abs(a->at(k,j)) >= TINY && flag == 1)
                {
                    mem = idx[i];
                    idx[i] = idx[k];
                    idx[k] = mem;
                    idx2 = k;
                    flag = 0;
                }
            }
            if (flag == 1)
            {
                for (k = 0; k < A->n_row; k++)
                {
                    if (std::abs(a->elem[A->n_col * k + j]) > TINY && std::abs(a->elem[A->n_col * i + k]) > TINY)
                    {
                        mem = idx[i];
                        idx[i] = idx[k];
                        idx[k] = mem;
                        idx2 = k;
                        flag = 0;
                    }
                }
            }
            if (idx2 == i)
            {
                printf("\n Singular matrix \n \n");
                a->elem[A->n_col * i + j] = TINY;
            }
            for (k = 0; k < A->n_row; k++)
            {
                mem = a->elem[A->n_col * i + k];
                a->elem[A->n_col * i + k] = a->elem[A->n_col * idx2 + k];
                a->elem[A->n_col * idx2 + k] = mem;
            }
        }
    }

    //------------------- Crout's algorithm for LU Decomposition -------------------
    for (j = 0; j < A->n_row; j++)
    {
        for (i = 0; i < A->n_row; i++)
        {
            if (i < j | i == j)
            {
                sum = a->elem[A->n_col * i + j];
                for (k = 0; k < i; k++)
                {
                    sum = sum - a->elem[A->n_col * i + k] * a->elem[A->n_col * k + j];
                }
                a->elem[A->n_col * i + j] = sum;
            }
            if (i > j)
            {
                sum = a->elem[A->n_col * i + j];
                for (k = 0; k < j; k++)
                {
                    sum = sum - a->elem[A->n_col * i + k] * a->elem[A->n_col * k + j];
                }
                a->elem[A->n_col * i + j] = sum / a->elem[A->n_col * j + j];
            }
        }
    }
    //---------------------------- Forward substituion -----------------------------
    for (k = 0; k < A->n_row; k++)
    {
        for (cont = 0; cont < A->n_row; cont++)
        {
            b[cont] = 0;
        }
        b[k] = 1;
        for (i = 0; i < A->n_row; i++)
        {
            sum = b[i];
            for (j = 0; j < i; j++)
            {
                sum = sum - a->elem[A->n_col * i + j] * x[j];
            }
            x[i] = sum;
        }
        //---------------------------- Backward substituion ----------------------------
        for (i = (A->n_row - 1); i >= 0; i--)
        {
            sum = x[i];
            for (j = i + 1; j < A->n_row; j++)
            {
                sum = sum - a->elem[A->n_col * i + j] * x[j];
            }
            x[i] = sum / a->elem[A->n_col * i + i];
        }
        for (cont = 0; cont < A->n_row; cont++)
        {
            R->elem[A->n_col * cont + idx[k]] = x[cont];
        }
    }
    delete a;
    b.clear();
    x.clear();
    idx.clear();

    return true;
}

#endif //LIBKF_HPP