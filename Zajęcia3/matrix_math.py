from typing import List
import numpy as np

def get_minor_mat(mat: np.ndarray, i: int, j: int) -> List[List[int]]:
    """
    Generate a minor mat of M for row 'i' and column 'j'.

    :param mat: mat to obtain a minor one by crossing out the row 'i' and
        the column 'j'
    :param i: index of row
    :param j: index of column

    :return: minor mat
    """
    mat_copy = np.delete(mat, i, 0)
    mat_copy = np.delete(mat_copy, j, 1)
        
    return mat_copy
    
def mat_determinant_assume_square(mat: List[List[int]]) -> int:
    determinant: int = 0
    
    if len(mat) == 1:
        return mat[0][0]

    y = 0
    line = mat[0]
    
    for x, cell in enumerate(line):
        minor = get_minor_mat(mat, y, x)
        determinant_unsigned = mat_determinant_assume_square(minor)
        determinant_of_minor = (-1) ** (y + x) * cell * determinant_unsigned
        determinant += determinant_of_minor
            
    return determinant   

def multiply(mat1: np.ndarray, mat2: np.ndarray):
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("Number of columns in mat1 must be equal to number of rows in mat2!")
    
    result = zeroes(mat1.shape[0], mat2.shape[1])
    
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            for k in range(mat2.shape[0]):
                result[i, j] += mat1[i, k] * mat2[k, j]
    
    return result

def transpose(mat: np.ndarray):
    transposed = np.array([[mat[j][i] for j in range(mat.shape[0])] for i in range(mat.shape[1])])
    return transposed

def zeroes(n: int, n2: int=None):
    result = np.ndarray((n, n if n2 is None else n2), dtype=float)
    result.fill(0.)
    
    return result

def identity(n: int):
    result = zeroes(n)
    
    for i in range(n):
        result[i][i] = 1
        
    return result

def inverse(mat: np.ndarray):
    n = mat.shape[0]
    if n != mat.shape[1]:
        raise ValueError("Matrix is not square!")
    
    extended_mat = np.hstack([mat, identity(n)])
    
    for col in range(n):
        max_row = max(range(col, n), key=lambda i: abs(extended_mat[i][col]))
        extended_mat[[col, max_row]] = extended_mat[[max_row, col]]
        
        extended_mat[col] /= extended_mat[col][col]
        
        for i in range(n):
            if i != col:
                extended_mat[i] -= extended_mat[col] * extended_mat[i][col]
                
    inverted_mat = extended_mat[:, n:]
    return inverted_mat