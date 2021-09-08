import operator, math, random, functools, itertools
import numpy as np
from scipy.sparse.linalg import splu
from rubik_solver import Cubie, utils, Move # thistlethwaite, kociemba
import pygame
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import sys
import keyboard
from pygameButton import button
import collections
import itertools

processMoves = lambda moves: list(itertools.chain(*[[movee[0] for i in range(int(movee[1].replace("'","3")))] if len(movee) is 2 else movee[0] for movee in moves]))

def det(matrix, triangular_method=False):
    if triangular_method:
        lu = splu(matrix) # it can be showen using cofactor expansion and induction that det of triangular is product of diag
        return reduce(lambda x, y: x*y, np.concatenate([lu.U.diagonal(),lu.L.diagonal()]))
    if matrix.shape == (2, 2):
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:    
        big = np.tile(matrix[1:], (len(matrix), 1, 1)) #big = np.repeat(matrix[None,:], repeats=3, axis=0)
        signs = np.vectorize(lambda indice: ((-1) ** (indice % 2)))(list(range(len(matrix)))) # starmap
        for (fc, height), value in np.ndenumerate(big[:, :, 0]):
            big[fc][height] = [20] + list(big[fc][height][0:fc]) + list(big[fc][height][fc+1:])
    big = big[:][:][big != 20].reshape(len(matrix),len(matrix)-1,len(matrix)-1) # in 3x3 matrix it will be 3,2,2
    return np.array([signs[i] * det(big[i]) for i in range(len(big))])
cross = lambda a,b: det(np.concatenate((np.ones(3),a,b),axis=0).reshape((3,3)))
sinTaylor = lambda theta: functools.reduce(lambda x, y: x+y, [((-1)**n)*(theta**(2*n+1))/math.factorial(2*n+1) for n in range(50)])
dot = lambda a,b: sum([i*j for (i, j) in zip(a, b)])

def rot(vector, axis, angle, matrix=False, angles=False): # rodrigues formula or chaining rotation matrices
    if matrix:
        rotationMat = lambda x,y,z: dict( [("x", np.array(( (1, 0, 0), (0, np.cos(x), -np.sin(x)), (0, np.sin(x),  np.cos(x)) ))),
            ("y", np.array(( (np.cos(y), 0, np.sin(y)), (0, 1, 0), (-np.sin(y),  0, np.cos(y)) ))),
            ("z", np.array(( (np.cos(z), -np.sin(z), 0), (np.sin(z),  np.cos(z), 0), (0, 0, 1) )) )])
        matmul = lambda m1,m2: [[sum(x * y for x, y in zip(m1_r, m2_c)) for m2_c in zip(*m2)] for m1_r in m1] # or einstein sum C[i][j] += A[i][k]*B[k][j]
        mat = rotationMat(*angles) # bruh remember not abelian you dumb cunt!!!
        return matmul(matmul(matmul([vector], mat["x"]), mat["y"]), mat["z"]) 
    else:    
        k = np.eye(3,dtype=int)[axis]                                          
        return vector*sinTaylor(math.pi/2+angle)+cross(k, vector)*sinTaylor(angle)+k*dot(k,vector)*(1-sinTaylor(math.pi/2+angle))

def rotate_matrix_ccw( m ):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1,-1,-1)]
def rotate_matrix_cw(matrix):
    return list(list(x)[::-1] for x in zip(*matrix))   

class Point:
    def __init__(self, *args):
        self.coor = self.convertedCoor = self.coorNoMove = np.array([*args]) 
    def convert(self, field, dist):
        coefficient = field / (dist + self.coor[2]) # z
        self.convertedCoor[0] = self.coor[0] * coefficient + 1000 / 2 # x , width
        self.convertedCoor[1] = -self.coor[1] * coefficient + 700 / 2 # y, height
        #newPoint = point(self.coor[0] * coefficient + 640 / 2, -self.coor[1] * coefficient + 480 / 2, self.coor[2])
        return self
    def rotate(self, axis, angle, basis=False):
        global cubeList
        #self.coor = rot(self.coor, axis, np.radians(angle))
        if basis == False:
            axis = np.eye(3)[axis]
        else:
            if axis == 0:
                axis = cubeList[13].center()-cubeList[12].center()
            elif axis == 1:
                axis = cubeList[4].center()-cubeList[22].center() # fine
            else:
                axis =  cubeList[13].center()-cubeList[10].center()       
            axis = axis / np.linalg.norm(axis)
        rot = Rotation.from_rotvec(np.radians(angle) * axis) ######################################## do we need to recreate this every time?
        self.coor = rot.apply(np.array(self.coor))
        if basis == False:
            self.coorNoMove = rot.apply(np.array(self.coorNoMove))
        return self    

class Cube:
    def __init__(self, moveAxis=None, dir=None, squaresPointCoors=None): # moveAxis and dir are lists 
        self.moveAxis, self.dir = moveAxis, dir
        self.resetPoints(squaresPointCoors)
        self.faces = [[0,1,2,3],[1,5,6,2],[5,4,7,6],[4,0,3,7],[0,4,5,1],[3,2,6,7]] # refering to indices
        self.colors = {} # {1: "red",2: "green"} means face 1 is red and face 2 is green   
    def resetPoints(self, squaresPointCoors): # moveAxis 0/1/2, dir=-1/1
        if squaresPointCoors:
            self.squaresPoint = [Point(*coor) for coor in squaresPointCoors]
        else:
            self.squaresPoint = [Point(-1,1,-1), Point(1,1,-1), Point(1,-1,-1), Point(-1,-1,-1), Point(-1,1,1), Point(1,1,1), Point(1,-1,1), Point(-1,-1,1)]
        if self.moveAxis is not None:
            for count, axis in enumerate(self.moveAxis):
                for point in self.squaresPoint:
                    point.coor[axis] += self.dir[count]
    def center(self):
        #return np.sum([point.coor for point in self.squaresPoint], axis=0) / 8
        return np.mean([point.coor for point in self.squaresPoint], axis=0)
                            
                       
    def move(self, angleX, angleY, angleZ = 0, moves=False, reverse=False):
        #self.resetPoints()
        transformed = []
        if moves:
            for count, point in enumerate(self.squaresPoint):
                transformed.append((point.rotate(2, angleZ+(-angleZ*2*reverse), basis=True).rotate(1, angleY+(-angleY*2*reverse), basis=True).rotate(0, angleX+(-angleX*2*reverse), basis=True).convert(400, 10)))
            return transformed
        angleX = angleX % 360
        angleY = angleY % 360
        for count, point in enumerate(self.squaresPoint):
            transformed.append((point.rotate(1, -(angleY-prevAngleY)).rotate(0, -(angleX-prevAngleX)).convert(400, 10)))
        return transformed

angle, prevAngleX, prevAngleY, angleX, angleY, prevPos, pos, start = 0, 0, 0, 0, 0, None, None, False 
pygame.init()
DISPLAY = pygame.display.set_mode((1000, 700))
clock = pygame.time.Clock()

buttonScramble = button(pygame.Color("red"), 0, 0, 100, 100, "Scramble")
buttonSolve = button(pygame.Color("blue"), 100, 0, 100, 100, "Solve")

cube1 = Cubie.Cube("yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww")
cube2 = Cubie.Cube("yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww")
cubeString = cube1.to_naive_cube(True)

shuffle_moves = [move.__str__().strip() for move in cube2.shuffle()]
       
colorsTranslation = {"r":"red", "b":"blue", "w": "white","o":"orange","g":"green","y":"yellow"}        

cubeList = [
    Cube([0,1,2], [-2, 2, 2]), Cube([0,1,2], [0, 2, 2]), Cube([0,1,2], [2, 2, 2]), Cube([0,1,2], [-2, 2, 0]), Cube([0,1,2], [0,2,0]), Cube([0,1,2], [2,2,0]), Cube([0,1,2], [-2,2,-2]), Cube([0,1,2], [0,2,-2]) , Cube([0,1,2], [2,2,-2]),
    Cube([0,1,2], [-2, 0, 2]), Cube([0,1,2], [0, 0, 2]), Cube([0,1,2], [2, 0, 2]), Cube([0,1,2], [-2, 0, 0]), Cube([0,1,2], [0,0,0]), Cube([0,1,2], [2,0,0]), Cube([0,1,2], [-2,0,-2]), Cube([0,1,2], [0,0,-2]) , Cube([0,1,2], [2,0,-2]),
    Cube([0,1,2], [-2, -2, 2]), Cube([0,1,2], [0, -2, 2]), Cube([0,1,2], [2, -2, 2]), Cube([0,1,2], [-2, -2, 0]), Cube([0,1,2], [0,-2,0]), Cube([0,1,2], [2,-2,0]), Cube([0,1,2], [-2,-2,-2]), Cube([0,1,2], [0,-2,-2]) , Cube([0,1,2], [2,-2,-2])
]

startScramble = False
startSolve = False

movesDict = {
    "U":{"pieces":[0,1,2,3,4,5,6,7,8], "color":"yellow", "faceIndex":4, "axis": 1}, # had to rotate 90 deg cw intuitive
    "L":{"pieces":[0,3,6,9,12,15,18,21,24], "color":"blue", "faceIndex":3, "axis": 0},
    "F":{"pieces":[6,7,8,15,16,17,24,25,26], "color":"red", "faceIndex":0, "axis":2},
    "R":{"pieces":[8,5,2,17,14,11,26,23,20], "color":"green", "faceIndex":1, "axis": 0},
    "B":{"pieces":[2,1,0,11,10,9,20,19,18], "color":"orange", "faceIndex":2, "axis": 2},
    "D":{"pieces":[24,25,26,21,22,23,18,19,20], "color":"white", "faceIndex":5, "axis": 1} # had to transpose intuitive
}


def initializeCube(cubeString, default_pos=True):
    if default_pos == False:
        listOfListOfPoints = [cube.squaresPoint for cube in cubeList] # list of lists of points
        listOfListsOfCoors = [[point.coorNoMove for point in listOfPoints] for listOfPoints in listOfListOfPoints]
        for i in range(len(cubeList)):
            cubeList[i] = Cube(squaresPointCoors=listOfListsOfCoors[i]) # don't need to reinitialize object 
            
    cube_faces = { k:cubeString[index*9:(index+1)*9] for index,k in enumerate("ybrgow") } 
    for key, value in movesDict.items():
        for pieceIndex, piece in enumerate(value["pieces"]):
            cubeList[piece].colors[value["faceIndex"]] = colorsTranslation[cube_faces[value["color"][0]][pieceIndex]]
    for i in range(len(cubeList)):            
        for faceNumber in range(6):
            if faceNumber not in list(cubeList[i].colors.keys()):
                cubeList[i].colors[faceNumber] = "black"  

#cube1.move(Move.Move("R"))
#cube1.move(Move.Move("F"))
#cubeString = cube1.to_naive_cube(True)
initializeCube(cubeString)

# for key, value in movesDict.items():
#     for piece in value["pieces"]:
#         cubeList[piece].colors[value["faceIndex"]] = value["color"]
# for i in range(len(cubeList)):            
#     for faceNumber in range(6):
#         if faceNumber not in list(cubeList[i].colors.keys()):
#             cubeList[i].colors[faceNumber] = "black"
        
# ''.join(collections.OrderedDict.fromkeys(cubeString).keys()) # distinct colors in string

angleAngle = 90
test_moves = ["R", "F", "D", "B", "F"]
move_index = 0
       
stop_sign = False

while True:    
    #print(basis_vectors)
    stop_sign = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if buttonScramble.click(event):
            startScramble = True
            test_moves = processMoves(shuffle_moves)
            move_index = 0
            
        if buttonSolve.click(event):
            startSolve = True
            test_moves = processMoves([move.__str__().strip() for move in utils.solve(cube2.to_naive_cube(True), method="Kociemba")])
            move_index = 0
            angleAngle = 90
    
    if pygame.mouse.get_pressed()[0] == 1:
        #print(pygame.mouse.get_pos())
        start = True
        if prevPos:
            pos = pygame.mouse.get_pos()
            deltaX, deltaY = pos[0] - prevPos[0], pos[1] - prevPos[1]
            angleX, angleY = prevAngleX + deltaY/3, prevAngleY + deltaX/3
            prevPos = pos
        else:
            prevPos = pygame.mouse.get_pos()
    else:
        if start == True:
            prevPos = None
    clock.tick(10)
    DISPLAY.fill((0,0,0))    
    
    transformedTotal = [cubeList[i].move(angleX, angleY) for i in range(len(cubeList))]
    
    if startScramble==True or startSolve==True:
        if angleAngle != 0:
            moveDict = movesDict[test_moves[move_index]]
            for number in moveDict["pieces"]:
                cubeList[number].move(*np.eye(3)[moveDict["axis"]]*15, moves=True, reverse=test_moves[move_index] in ["D", "L", "B"])   
            angleAngle = angleAngle - 15
        else:
            if move_index is not len(test_moves)-1: 
                move_index += 1   
                angleAngle = 90
                cube1.move(Move.Move(test_moves[move_index-1]))
                cubeString = cube1.to_naive_cube(True)
                initializeCube(cubeString,default_pos=False)
                stop_sign = True
            else:
                cube1=cube2  
                                       
    prevAngleX = angleX
    prevAngleY = angleY
    
    average_z_list = []
    
    if not stop_sign:
        for cubeIndex, cube in enumerate(cubeList):
            for faceIndex, face in enumerate(cube.faces):
                z = [transformedTotal[cubeIndex][face[i]].coor[2] for i in range(len(face))]
                average_z = sum(z) / 4
                average_z_list.append({"index": faceIndex, "avg": average_z, "cubeIndex": cubeIndex})
                
        average_z_list.sort(key = lambda dicti: dicti["avg"], reverse=True)
        for dictionary in average_z_list:
            face = cubeList[dictionary["cubeIndex"]].faces[dictionary["index"]] # doesn't matter 
            pointList = [transformedTotal[dictionary["cubeIndex"]][face[i]].convertedCoor[:-1] for i in range(4)]
            pygame.draw.polygon(DISPLAY, pygame.Color(cubeList[dictionary["cubeIndex"]].colors[dictionary["index"]]), pointList)
    
    
    buttonScramble.draw(DISPLAY)
    buttonSolve.draw(DISPLAY)

    if not stop_sign:
        pygame.display.flip()
        pygame.display.update()      
       
        
# remember angle = arccos((trace(R)-1)/2)
#[getattr(Cube, move)(c) for move in [random.choice(["U","D","F","B","R","L"]) for _ in range(1)]]
#print(utils.solve(''.join(c._color_list()).lower(), 'Kociemba'))
#print(rot(np.array([1,5,1]),0,math.radians(90)))
#print(rot(np.array([1,5,1]),0,0,matrix=True, angles=(math.radians(270), 0, 0)))        
        
        
