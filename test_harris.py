import cv2
import numpy as np
import functools
from scipy import ndimage
from scipy.linalg import lu
# How does one nigger compute the determinant using laplace cofactor expansion??? 🤔    
def recursiveTrauma(matrix, triangular_method=False): # default arg total 0
    if triangular_method: #np.all(matrix==0)
        U, L = lu(matrix)[1:] # matrix is product of upper triangular and lower
        # it can be showen using cofactor expansion and induction that det of triangular is product of diag
        return functools.reduce(lambda x, y: x*y, np.concatenate([U.diagonal(),L.diagonal()]))
    if matrix.shape == (2, 2):
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:    
        big = np.tile(matrix[1:], (len(matrix), 1, 1)) #big = np.repeat(matrix[None,:], repeats=3, axis=0)
        signs = np.vectorize(lambda indice: ((-1) ** (indice % 2)))(list(range(len(matrix)))) # starmap
        for (fc, height), value in np.ndenumerate(big[:, :, 0]):
            big[fc][height] = [0] + list(big[fc][height][0:fc]) + list(big[fc][height][fc+1:])
            # NOT TRUE: 0 placeholder (always 1 zero) without it size and dimension do not match
            # ended up making numpy object (varying axis not homogenous)
            # ok so in the end it was true or at least I don't know how to make objects
    total = 0    
    big = big[:][:][big != 0].reshape(len(matrix),len(matrix)-1,len(matrix)-1) # in 3x3 matrix it will be 3,2,2
    for i in range(len(big)):
        total += signs[i] * matrix[0][i] * recursiveTrauma(big[i])
    return total
# why are you so slow 😔
def HarrisCorners(image, threshold):
    iy, ix = np.gradient(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    ixx, iyy, ixy = ix*ix, iy*iy, ix*iy
    result, max = np.zeros(image.shape), 0
    ixx, iyy, ixy = list(map(functools.partial(ndimage.gaussian_filter, sigma=2), [ixx,iyy,ixy]))    
    for y in range(image.shape[0]):
        for x in range(image.shape[0]):
            m = np.array([[ixx[y,x], ixy[y,x]], [ixy[y,x], iyy[y,x]]], dtype=np.float64)
            calc = recursiveTrauma(m, True) - 0.04 * np.power(np.trace(m), 2)
            #calc = np.linalg.det(m) - 0.04 * np.power(np.trace(m), 2)
            if calc > threshold:
                image[y,x] = (0,255,255)
    return image   
if __name__ == '__main__':
    my_matrix = np.array([[8,-7],[2,1]]) 
    # print(lu(my_matrix).L)
    # print("nigger")
    # print(lu(my_matrix).U)
    cv2.imshow("result", HarrisCorners(cv2.imread("harris_chess.jpg"), 10000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
