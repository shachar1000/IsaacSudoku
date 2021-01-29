import numpy as np
from matplotlib import pyplot as plt
import itertools
import collections
import cv2
import time
from numpy.fft import fft2, ifft2
from matplotlib import pyplot as plt
import math
from scipy.sparse.linalg import splu
from functools import reduce
import imutils
from sklearn.cluster import KMeans
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
import operator
# import joblib
# from skimage.feature import hog
from imutils.perspective import four_point_transform
import pytesseract
import random
#clf = joblib.load("digits_cls.pkl")
from collections.abc import Iterable
from functools import partial
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.neighbors import KDTree

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def generalized_convolution(matrix, kernel_arg, blur_coefficient=3, sigma=1, method_3d=False, method_seperate=True):
    xx, yy = np.meshgrid(np.linspace(-1, 1, blur_coefficient), np.linspace(-1, 1, blur_coefficient))
    gaussian_kernel = np.exp(-(xx ** 2 + yy ** 2)/(2*sigma**2)) * (1 / (2.0 * np.pi * sigma**2)) # sigma is standard deviation
    kernels_dict = {"gaussian": gaussian_kernel, "shift": np.array([[0,0,0], [1,0,0], [0,0,0]])}
    kernel = kernel_arg if isinstance(kernel_arg, collections.abc.Iterable) and not isinstance(kernel_arg, str) else kernels_dict[kernel_arg]
    result = np.zeros(matrix.shape, dtype=np.int)
    print(gaussian_kernel)
    if matrix.ndim == 2:
        super_matrix = np.zeros((matrix.shape[0] + (kernel.shape[0] // 2)*2, matrix.shape[1] + (kernel.shape[1] // 2)*2), dtype=np.int)
        super_matrix[1:-1, 1:-1] = matrix
        cartesian_product = np.transpose([np.tile([], len([])),np.repeat([],len([]))]) # 0101*0011
        cartesian_product = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)] # create indice pairs using non-general cartesian product
        for (r, c), value in np.ndenumerate(matrix): #multidimensional index iterator
            neigh = list(itertools.islice(itertools.starmap((lambda y_offset, x_offset: super_matrix[r+y_offset+1][c+x_offset+1]), cartesian_product), 0, None))
            result[r, c] = np.sum(np.concatenate(np.einsum("ij, ij -> ij", np.array(neigh).reshape((3, 3)), kernel))) # can also multiply value from enumerate by value of kernel at (r2,c2) indices
            # concatenate because sum is along axis, hamdard product 
    elif matrix.ndim == 3: # use pad method, margin of shape//2 except on z axis, instead of ndim we can do len(shape)
        print(matrix.shape)
        k_h, k_w = kernel.shape[0], kernel.shape[1]
        #super_matrix = np.pad(matrix, pad_width=((k_h//2,k_h//2),(k_w//2,k_w//2),(0,0)), mode='constant', constant_values=0).astype(np.float32)
        def recursive_product(*args): #################
            return [[x] + p for x in args[0] for p in recursive_product(*args[1:])] if args else [[]]
        generalized_cartesian_product = recursive_product(*([[-1, 0, 1]]*matrix.ndim))
        h, w = k_h // 2, k_w // 2
        if method_3d:
            super_matrix = np.pad(matrix, pad_width=((k_h//2,k_h//2),(k_w//2,k_w//2), (0, 0)), mode='constant', constant_values=0).astype(np.uint8)
            for (r, c), value in np.ndenumerate(matrix[:, :, 0]): # splice 3rd dimension
                neigh_z = list(itertools.islice(itertools.starmap((lambda z_offset, y_offset, x_offset: super_matrix[r+y_offset+1][c+x_offset+1][1+z_offset]), generalized_cartesian_product), 0, None))
                #print(np.array(neigh_z).reshape((3,3,3)))
                #neigh_y = super_matrix[1+r-h:1+r-h+k_h, 1+c-w:1+c-w+k_w] # ndenumerate matrix is same as nested loop super matrix -h -w + 1 + 1
                # apparently incorrect interpretation of 3d matrices. The neigh_y (y written in retrospect) 3x3x3 matrix is 3 matrices sorted by y each containing 3 arrays sorted by x and elements sorted by z
                # and not 3 matrices sorted by z as thought before.
                # we want to take yxz matrix and find 3 matrices, in which z values are constant for each of them
                # it turns out that's what the "incorrect interpretation does exactly that. (because cartesian product is z->y->x)
                for rgb in range(matrix.shape[matrix.ndim-1]): # k  k rgb                                               #z
                    result[r, c, rgb] = np.sum(np.concatenate(np.einsum("ij, ij -> ij", np.array(neigh_z).reshape((3, 3, 3))[rgb], kernel)))
        elif method_seperate: ######### instead of thinking in 3 dimensions, seperate to matrix.shape[last_dim] seperate 2d convolution (this can be applied recursively to N dimensions)
            def convolution_inner(matrix, kernel): #the reason we don't do recursion with the same function is that neighbors will not work with k>3
                result_inner = np.zeros(matrix.shape, dtype=np.int)
                super_matrix_inner = np.pad(matrix, pad_width=((k_h//2,k_h//2),(k_w//2,k_w//2)), mode='constant', constant_values=0).astype(np.uint8)
                start_time = time.time()
                for (r, c), value in np.ndenumerate(matrix): # 0 to matrix = h to super_matrix.shape[]-h
                    result_inner[r][c] = (super_matrix_inner[r:r+k_h, c:c+k_w].flatten()*kernel.flatten()).sum()
                print("--- %s seconds per convolution---" % (time.time() - start_time))
                return result_inner
            for rgb in range(3):
                result[:, :, rgb] = convolution_inner(matrix[:, :, rgb], kernel)
    return result.astype(np.uint8)

# One big brain nigga named Fourier found a neat method to compute convolution using FFT
#convolution = np.real(ifft2(fft2(matrix)*fft2(kernel, s=matrix.shape)))

def roll(array, rollN):
    return list(itertools.islice(itertools.cycle(array),rollN,rollN+len(array)))
    # unshift using matrix diagonal sum
    # np.sum(np.diag(the_array,1),0)[:-1]
    
# compute using shoelace formula which is a special case of Green's theorem 
# 0.5 sigma xi*yi+1-xi+1*yi (same as det)
def contourArea(contours): # numerical integration
    # integral of xdy-ydx (where curl of vector field is 1)
    contours = np.array([np.array(weird[0]) for weird in contours])
    x, y = contours[:,0], contours[:,1]
    area = 0.5*np.sum(y[:-1]*difference(x)-x[:-1]*difference(y))
    return np.abs(area)
    
def difference(array):
    diff_list = []
    for a, b in zip(array[0:], array[1:]): 
        diff_list.append(b-a) 
    return np.array(diff_list)
    
# How does one nigger compute the determinant using laplace cofactor expansion??? 🤔    
def recursiveTrauma(matrix, triangular_method=False): # default arg total 0
    if triangular_method:
        lu = splu(matrix) # matrix is product of upper triangular and lower
        # it can be showen using cofactor expansion and induction that det of triangular is product of diag
        return reduce(lambda x, y: x*y, np.concatenate([lu.U.diagonal(),lu.L.diagonal()]))
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
# https://www.qc.edu.hk/math/Advanced%20Level/Point_to_line.htm    
def ramer_douglas_peucker(pts, epsilon):
    max_dist, index = 0, 0 
    pts = [np.array(pt) for pt in pts] # this line probably saved me in cv2 contours
    for i in range(1, len(pts)-1): # skip first and last
        dist_pt_line = np.linalg.norm(np.cross(pts[-1]-pts[0], pts[i]-pts[0]))/np.linalg.norm(pts[-1]-pts[0]) # can also use np.sqrt(x.dot(x))
        if dist_pt_line > max_dist:
            max_dist = dist_pt_line
            index = i
    if max_dist > epsilon: # subdivide line on index of point with largest perpendicular distance
        return ramer_douglas_peucker(pts[0:index+1], epsilon)[:-1]+ramer_douglas_peucker(pts[index:], epsilon)    
    else:
        return [pts[0], pts[-1]] 
# why are you so slow 😔
def HarrisCorners(image, threshold):
    iy, ix = np.gradient(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    ixx, iyy, ixy = ix*ix, iy*iy, ix*iy
    result, max = np.zeros(image.shape), 0
    ixx, iyy, ixy = list(map(functools.partial(ndimage.gaussian_filter, sigma=2), [ixx,iyy,ixy]))    
    for y in range(image.shape[0]):
        for x in range(image.shape[0]):
            m = np.array([[ixx[y,x], ixy[y,x]], [ixy[y,x], iyy[y,x]]], dtype=np.float64)
            calc = recursiveTrauma(m) - 0.04 * np.power(np.trace(m), 2)
            #calc = np.linalg.det(m) - 0.04 * np.power(np.trace(m), 2)
            if calc > threshold:
                image[y,x] = (0,255,255)
    return image            

# epsilon = 0
# x_ = np.arange(0,5,0.01)        
# y_ = [math.exp(-x)*math.cos(2*math.pi*x) for x in x_]        
# pairs = [[x_[i], y_[i]] for i in range(len(y_))]       
# for i in range(100):
#     epsilon += 0.001
#     x_, y_ = np.split(ramer_douglas_peucker(pairs, epsilon),[-1],axis=1)
#     plt.plot(x_, y_)
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()
# 

#M = cv2.getPerspectiveTransform(rect, dst)
#warped = cv2.warpPerspective(img, M, (max_width + 10, max_height + 10))        
        
def find_board(image, desired_size=800, other_size=450):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7),3) 
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    cnts = list(filter(lambda x: contourArea(x) > 100, cnts))  
    # sort the cnts according to area 
    for c in list(sorted(cnts, key=contourArea, reverse=True)):
        simplified = ramer_douglas_peucker(c, 0.01 * cv2.arcLength(c, True))
        simplified = np.array([np.array(point) for point in simplified])
        if len(simplified) in [4,5]:
            oneContourToRuleThemAll = simplified
            break
    output = image.copy()
    mask = np.zeros(image.shape, np.uint8)               # draw green (0, 255, 0), 2                
    cv2.drawContours(mask, [oneContourToRuleThemAll], -1, (255,255,255), -1)
    final =  cv2.bitwise_and(image, mask)
    quad_contour = cv2.approxPolyDP(oneContourToRuleThemAll, 0.1 * cv2.arcLength(oneContourToRuleThemAll, True), True)
    #warped = four_point_transform(final, quad_contour.reshape(4,2))
    # h, status = cv2.findHomography(quad_contour.reshape(4,2), np.float32([(image.shape[1], 0),(0,0),(0, image.shape[0]), image.shape[:-1][::-1]]))
    # warped = cv2.warpPerspective(final, h, image.shape[:-1][::-1])
    # shoudln't be in accordance with shape of image but 1:1 aspect ratio
    size, desired = other_size, desired_size
    h, status = cv2.findHomography(quad_contour.reshape(4,2), np.float32([(size, 0),(0,0),(0, size), (size, size)]))
    warped = cv2.warpPerspective(final, h, (size, size))
    cv2.drawContours(final,[quad_contour],0,(0,0,255),2)
    ht, wd, cc= warped.shape
    result = np.full((desired,desired,cc), (0,0,0), dtype=np.uint8)
    xx = (desired - wd) // 2
    yy = (desired - ht) // 2
    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = warped
    return {"final":final, "warped":warped, "warped_padding":result, "homography":h}

def pointLineIntersection(l1p1, l1p2, l2p1, l2p2):
    fixed = np.hstack((np.vstack([l1p1,l1p2,l2p1,l2p2]), np.ones((4, 1)))) # add 1 for z cord    
    line1, line2 = np.cross(fixed[0], fixed[1]), np.cross(fixed[2], fixed[3])
    x, y, z = np.cross(line1, line2)
    return (x/z, y/z)
    
def mask_poly(image, corners):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array(corners, dtype=np.int32), (255,)*image.shape[2])
    masked_image = cv2.bitwise_and(image, mask)
    first = corners[0][0]
    last = corners[0][2]     # 01
                             # 32
    return_val = masked_image[first[1]:last[1], first[0]:last[0]].copy() 
    #cv2.imshow("return_val", return_val)
    return return_val 

# hough sinusoid bois intersection 〰️  
def hough_transform(image, cannied_image, thresh):
    half_height, half_width = np.array(image.shape[:2]) / 2
    diagonal = math.sqrt((half_height*2)**2 + (half_width*2)**2)
    rhos = np.arange(-diagonal, diagonal, 2*diagonal/180)
    thetas = np.deg2rad(np.arange(0, 180, 1)) # p (rho) = xcosa+ysina which can be achieved using matrix multiplication
    accumulator = np.zeros((len(rhos), len(rhos)))                                                      # HUGE BUG: sin and then cos
    rhos_calc = np.matmul(np.argwhere(cannied_image != 0) - np.array([[half_height, half_width]]), np.array([np.sin(thetas), np.cos(thetas)]))
    print(len(rhos_calc))
    # argwhere is indices (x,y) where ...
    for point in rhos_calc:
        for theta_index in range(len(point)):
            # find index of closest rho value to what we calculated
            rho_calc = point[theta_index]
            rho_index = np.argmin(np.abs(rhos - rho_calc)) 
            accumulator[rho_index][theta_index] += 1
    # now convert accumulator to lines...
    lines = np.argwhere(accumulator > thresh)

    for line in lines: # extract line from polar ccordinates...
        # line is (rho_index), (theta_index)
        print(line)
        theta = thetas[line[1]]
        rho = rhos[line[0]]
        a = np.cos(theta)
        b = np.sin(theta)
        x =  a * rho + half_width
        y =  b * rho + half_height
        x1 = int(x + 1000 * (-b))
        y1 = int(y + 1000 * a)
        x2 = int(x - 1000 * (-b))
        y2 = int(y - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0))
matrix_sudoku = [[ 0 for i in range(9)] for j in range(9)]

def extraction(cell):
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh) # remove border for digit recognition
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        gray_cell = cv2.bitwise_and(thresh, thresh, mask=mask)
    return {"cell_img": cv2.cvtColor(gray_cell, cv2.COLOR_GRAY2BGR), "cntLen": len(cnts)}
    
def lines(image):
    global matrix_sudoku
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgy = cv2.Canny(gray, 40, 60, apertureSize=3)
    kernel_dilate = np.ones((3,3), np.uint8)
    kernel_erode = np.ones((3,3), np.uint8)
    edgy = cv2.dilate(edgy, kernel_dilate, iterations=2)
    edgy = cv2.erode(edgy, kernel_erode, iterations=2)
    #hough_transform(image, edgy, thresh=220)
    lines = [excess[0] for excess in cv2.HoughLines(edgy,1,np.pi/180,350)]
    #lines = KMeans(n_clusters=20, random_state=0).fit(lines).cluster_centers_    
    duplicates = {i:[] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if (i==j): continue                   # rho, theta thresholds
            if (np.abs(lines[i]-lines[j])<np.array([25, 1])).all():
                duplicates[i].append(j)
    for key, value in dict((key,value) for key, value in duplicates.items() if value).items():
        points = [lines[key],*[lines[point] for point in value]]
        lines[key] = [sum(x)/len(x) for x in zip(*points)] # average rho, theta
    print(duplicates)
    #### filter lines
    
    # line segments points
    ls_p, angles = [], []
    for line in lines:
        rho,theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        ls_p.append([x1, y1, x2, y2])
        angles.append(abs(math.atan2(y2-y1, x2-x1) * 180 / math.pi))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    ls_p = np.array(ls_p)
    angles = np.array(angles).astype(int)
    def split(array, condition):
        return (array[condition], array[~condition])
    horizontal, vertical = split(ls_p, (angles < 10)) # if the sudoku is too rotated this can become a huge problem (0 intersections) originally 10
    # now for some vectorization magic
    #cartesian_product = np.transpose([np.tile(ls_p, len(ls_p)),np.repeat(ls_p,len(ls_p))])
    cartesian_product = list(itertools.product(horizontal, vertical)) # calculate intersection of horizontal line with each one of the vertical lines.
    intersections = np.array([pointLineIntersection(*np.array_split(np.array(L1L2).reshape((4, 2)), 4)) for L1L2 in cartesian_product])
    # point line intersection accepts 4 [x, y] and L1L2 is [[x,y,x,y],[x,y,x,y]]
    tree = KDTree(intersections)
    all_nn_indices = tree.query_radius(intersections, r=10)
    all_nns = [[intersections[idx] for idx in nn_indices] for nn_indices in all_nn_indices]
    all_nns = [nnn[0] for nnn in all_nns]
    intersections = np.unique([tuple(row) for row in all_nns], axis=0)
    for point in intersections: # intersections
        cv2.circle(image, tuple(np.array(point).astype(int)), 5, (0, 255, 0), 2)
    # first get rid of similar points maybe using kdtree (spatial data structure knn)
    # HOW THE FUCK does someone even come up with these algos jesus christ........... the value for the original is 40
    #intersections = {((x - (x % d)), (y - (y % d))) : (int(x),int(y)) for x, y in intersections}.values()
    # the idea is that dictionary keys are exclusive so similar points will overwrite  
    matrix = np.array(sorted(intersections , key=lambda x: x[1])).reshape(10,10,2)
    matrix = np.array([np.array(sorted(row, key=lambda x: x[0])) for row in matrix])

    final_image = np.zeros((image.shape[0]+500,image.shape[1]+500,3), np.uint8)
    x_offset = 100
    y_offset = 100
    centroids = [i for i in range(81)]
    print(matrix)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    for y in range(len(matrix)-1):
        for x in range(len(matrix[0])-1):
            corners = matrix[y:y+2,x:x+2].flatten().reshape(4,2)
            corners[2:4] = corners[2:4][::-1] # otherwise we get a hourglass
            centroids[x + 9*y] = np.mean(corners, axis=0)
            corners = np.int32([corners])
            #cv2.imshow("{},{}".format(y,x), mask_poly(image, corners))
            cell = cv2.resize(mask_poly(image, corners), (80, 80), interpolation=cv2.INTER_AREA)
            extraction_fc = extraction(cell.astype('uint8'))
            # kernel = np.ones((3,3),np.uint8)
            # digit = cv2.dilate(extraction1[0],kernel,iterations = 1) 
            # invert because we want black digit on white background
            digit_image = cv2.bitwise_not(extraction_fc["cell_img"]) if extraction_fc["cntLen"] > 0 else extraction_fc["cell_img"] 
            if extraction_fc["cntLen"]:
                #cv2.imshow("{},{}".format(y,x), digit_image)
                try:
                    txt = pytesseract.image_to_string(digit_image, config='--psm 6 --oem 3 outputbase digits')
                    matrix_sudoku[y][x] = int(txt[0])
                except:
                    matrix_sudoku[y][x] = 0  
            else:
                matrix_sudoku[y][x] = 0    
            final_image[y_offset*y:y_offset*y+digit_image.shape[0], x_offset*x:x_offset*x+digit_image.shape[1]] = digit_image            
            
    final_image = cv2.resize(final_image[:y_offset*y+digit_image.shape[0],:x_offset*x+digit_image.shape[1]], image.shape[:-1][::-1])            
    # need to reverse specific axis in matrix (x axis)
    # matrix_fixed_ordering = np.flip(matrix, (1,0))
    # print(matrix_fixed_ordering)    
    # print("nigger")
    # print(len(cartesian_product))    
    return {"centers":np.array(centroids),"lines_image":image, "final_image":final_image}
    
image = cv2.imread("self_sudoku_new_orient.jpeg")
#image = cv2.imread("sudoku.jpg")
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
only_board = find_board(image)       
lines_output = lines(only_board["warped_padding"].copy())
centers = lines_output["centers"]
centers[np.where(np.array(matrix_sudoku).flatten() != 0)] = 0

def get_default_args(func):
    signature = inspect.signature(func)
    return { k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty }
desired_size, other_size = get_default_args(find_board)["desired_size"], get_default_args(find_board)["other_size"]

scale_h, _ = cv2.findHomography(np.float32([(other_size, 0),(0,0),(0, other_size), (other_size, other_size)]), np.float32([(desired_size, 0),(0,0),(0, desired_size), (desired_size, desired_size)]))
centers = cv2.perspectiveTransform(centers.reshape(-1,1,2).astype(np.float32), scale_h).reshape(-1, 2)
print(matrix_sudoku)
#print(centers)
clicked = None
def process_click(event,x,y,flags,params, start, end):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > start[1] and y < end[1] and x > start[0] and x < end[0]:   
            print("Niggaaaaa")
            clicked = True
def solve():
    global matrix_sudoku
    global clicked
    for y in range(9):
        for x in range(9):
            if matrix_sudoku[y][x] == 0:
                for n in range(1, 10):
                    if possible(y, x, n):
                        matrix_sudoku[y][x] = n
                        k=(cv2.waitKey(250) & 0XFF) if not clicked else None
                        show(only_board["warped"])
                        solve()
                        matrix_sudoku[y][x] = 0

                return
    print(matrix_sudoku)            
    show(only_board["warped"], done=True)
    

def possible(y, x, n):
    global matrix_sudoku
    x0, y0 = (x // 3) * 3, (y // 3) * 3
    for X in range(x0, x0 + 3):
        for Y in range(y0, y0 + 3):
            if matrix_sudoku[Y][X] == n:
                return False  
    return n not in [[matrix_sudoku[j][i] for j in range(len(matrix_sudoku))] for i in range(len(matrix_sudoku[0]))][x] and n not in matrix_sudoku[y]

def show(only_board, done=False):
    first = centers[1]
    print(first)
    img_new = only_board.copy()
    factor = lines_output["lines_image"].shape[0]/img_new.shape[0] # homography with padding / without
    img_new = np.concatenate((cv2.resize(img_new, (0,0), fx=factor, fy=factor), lines_output["lines_image"], lines_output["final_image"]), axis=1)
    for count, center in enumerate(centers):
        if isinstance(center, Iterable): # if not 0
            x_i, y_i = count % 9, count // 9
            img_new = cv2.putText(img_new,str(matrix_sudoku[y_i][x_i]),tuple((center-300).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.8,(50, 0, 215),2,cv2.LINE_AA)       
    cv2.rectangle(img_new, (int(only_board.shape[1]*factor), 0), (int(only_board.shape[1]*factor+100), 100), (255, 0, 0), -1)        
    cv2.imshow('Please work',img_new) # bind arguments 
    cv2.setMouseCallback('Please work',partial(process_click, start=(only_board.shape[1]*factor, 0), end=(only_board.shape[1]*factor+100, 100)))
    if done:
        cv2.waitKey()  
    
# while True:
#     show(only_board)
#     k=cv2.waitKey(100) & 0XFF
#     if k== 27:
#         break
# cv2.destroyAllWindows()

solve()



# 
# def sudoku_solver():
#     pass
# 
# #cv2.imshow("test", mask_poly(image, [[(0,0),(0,100),(100,100),(100,0)]]))    
# 
# print(pointLineIntersection([0, 0],[2, 2],[0, 2],[2, 0]))
# 
# # gray = cv2.cvtColor(only_board, cv2.COLOR_BGR2GRAY)
# # blur = cv2.GaussianBlur(gray, (3,3),1)
# # 
# # edge = cv2.Canny(blur, 100, 200)
# # 
# # edge = cv2.dilate(
# #     edge,
# #     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
# #     iterations=1
# # )
# # edge = cv2.erode(
# #     edge,
# #     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
# #     iterations=1
# # )
# # cv2.imshow("edge", edge)
# #hough_transform(image, edge, thresh=400)
# #cv2.imshow("square.png", image)
# 
# 
# 
# 
# matrix = np.array([[11,2,3,4],
#                   [5,6,7,86],
#                   [9,10,111,12],
#                   [13,14,155,16]])
# print(recursiveTrauma(matrix, triangular_method=True))
# 
# 
# # shifted_identity_kernel = np.array([[0,0,0],
# #                    [1,0,0],
# #                    [0,0,0]])
# #img = cv2.imread('chess.jpeg')
# #img_blurred = generalized_convolution(img, "gaussian", blur_coefficient=3, method_3d=True, method_seperate=False)
# #print(generalized_convolution(matrix, "shift"))
# 
# def recursive_product(*args):
#     return [[x] + p for x in args[0] for p in recursive_product(*args[1:])] if args else [[]]
# 
# 
# print(recursive_product([-1, 0, 1],[-1, 0, 1],[-1, 0, 1]))


#cv2.imshow('image',img)
#cv2.imshow('image_blurred',img_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
