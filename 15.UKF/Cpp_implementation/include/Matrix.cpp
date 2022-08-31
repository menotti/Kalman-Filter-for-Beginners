

#include "Matrix.hpp"

//Check Symmetry
template<typename T>
bool Matrix<T>::is_symmetric()
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
template<typename T>
bool Matrix<T>::is_square()
    {
        return n_row == n_col;
    }


//Zero Initialization
template<typename T>
void Matrix<T>::zeros(){
        std::memset(elem.data(), T{0}, sizeof(elem));
    }

//Eye Matrix.. 
template<typename T>
void Matrix<T>::eye(T alpha)
{   
    for (int j = 0; j < n_col; j++){
        for(int i=0; i < this->n_row; i++)
            if(i == j) this->at(i,j) = alpha;
            else this->at(i,j) = T{0};
    }
}



//Creates a copy of matrix A
template<typename T>
void Matrix<T>::copy_of(Matrix<T> &A)
    {
        n_row = A.n_row;
        n_col = A.n_col;
        this->elem.clear();
        elem = A.elem;
    }


template<typename T>
void Matrix<T>::display()
{
    std::cout <<"Display Matrix\n";
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
        std::cout<< this->at(i,j) << " ";
        }
        std::cout<< "\n";
    }
    std::cout<< "\n";
}

