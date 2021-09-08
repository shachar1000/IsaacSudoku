matrix = [[0,1,2],[3,4,5],[6,7,8]]

def rotate_matrix_ccw( m ):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1,-1,-1)]
def rotate_matrix_cw(matrix):
    return list(list(x)[::-1] for x in zip(*matrix))    
    
print(rotate_matrix_cw(matrix))    





import itertools


moves = []
moves = ["F2", "R", "L2", "R'"]

processMoves = lambda moves: list(itertools.chain(*[[move[0] for i in range(int(move[1].replace("'","3")))] if len(move) is 2 else move[0] for move in moves]))
print(processMoves(moves))
