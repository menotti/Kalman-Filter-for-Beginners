#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream> 
#include <cstring>


template< class T>
class Matrix
{
    // private:
    // protected:
    public:
    std::vector<T> elem;
    int n_row, n_col;
    
    public:
    // int col() const {{return n_col;}}
    // int row() const {{return n_row;}}
    // Returns size matrix
    Matrix(int row, int col)
    {
        n_col = col;
        n_row = row;
        elem.resize(n_col * n_row, 0.0);
    }

    int size() { return elem.size(); }

    // Constructor

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

    void eye(T alpha);
    
    void getT();

    void copy_of(Matrix &A);

    bool equal_dim(Matrix &B){
        return ((n_row == B.n_row) && (n_col == B.n_col));}
    
    //Pointers & address intructions...
    T &at(int i, int j){return this->elem[n_row*i+j];}
    double *data() {return elem.data();}


    void display();
};



#endif //LIBKF_HPP