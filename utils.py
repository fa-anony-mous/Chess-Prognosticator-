# These are all the functions we used for performing chess piece detection and chessboard detection.

# Import Necessary Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from ultralytics import YOLO


# -----------------------------------------------------------------------------------------------------------------
# Find the Hough lines for the uploaded chessboard and return the outlined image.
# -----------------------------------------------------------------------------------------------------------------

def find_Hough_Lines(img_path):

    img = cv2.imread(img_path)

    # Set the lower and upper thresholds for Canny Edge detection.
    t_lower = 250  # Lower Threshold 
    t_upper = 256  # Upper threshold 

    slant_img = img.copy()
    edges_slant = cv2.Canny(slant_img, t_lower, t_upper) # Canny Edge detection.
    lines = cv2.HoughLines(edges_slant,1,np.pi/360,250) # Get the Hough lines.
    w = img.shape[1]

    # Draw the Hough lines on the I/P image.
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + w*(-b))
            y1 = int(y0 + w*(a))
            x2 = int(x0 - w*(-b))
            y2 = int(y0 - w*(a))
            cv2.line(slant_img,(x1,y1),(x2,y2),(0,255,0),1)
    
    # Return the outlined image.
    return slant_img


# -----------------------------------------------------------------------------------------------------------------
# Find the square boundaries and get the dividing lines.
# -----------------------------------------------------------------------------------------------------------------

def find_Square_Boundaries(slant_img):
    
    prev_x = 0
    prev_y = 0
    x_coords = []
    y_start_coords = []
    y_end_coords = []
    
    # Find all the lines running paralel to x-axis and at an offset of 100px from each other.
    for i in range(slant_img.shape[0]):
        if(slant_img[i,0,0]==0 and slant_img[i,0,1]==255 and slant_img[i,0,2]==0 and i-prev_x > 100):
            x_coords.append(i)
            prev_x = i

    x_coords = x_coords[:9]

    # Do the same for the lines running along the chessboard columns at the top.
    for i in range(slant_img.shape[1]):
        if(slant_img[0,i,0]==0 and slant_img[0,i,1]==255 and slant_img[0,i,2]==0 and i-prev_y > 100):
            y_start_coords.append(i)
            prev_y = i

    # Do the same for the lines running along the chessboard columns at the bottom.
    prev_y = 0 
    for i in range(slant_img.shape[1]):
        if(slant_img[x_coords[-1]+30,i,0]==0 and slant_img[x_coords[-1]+30,i,1]==255 and slant_img[x_coords[-1]+30,i,2]==0 and i-prev_y > 150):
            y_end_coords.append(i)
            prev_y = i
    
    # Return the line coordinates.
    return (x_coords, y_start_coords, y_end_coords)


# -----------------------------------------------------------------------------------------------------------------
# Find the perspective transformation of the chessboard  using the line coordinates from the above function.
# -----------------------------------------------------------------------------------------------------------------

def find_Perspective_Transform(img_path, x_coords, y_start_coords, y_end_coords):

    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]
    
    src_points = np.float32([[y_start_coords[0]-15, x_coords[0]-15], [y_start_coords[-1]+60, x_coords[0]], [y_end_coords[0], x_coords[-1]], [y_end_coords[-1]+20, x_coords[-1]]])
    dst_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))

    return img_output, projective_matrix


# -----------------------------------------------------------------------------------------------------------------
# Find the chessboard centers by dividing the perspective image into a 16 x 16 grid.
# -----------------------------------------------------------------------------------------------------------------

def find_ChessBoard_Centers(img_path):
    
    img = cv2.imread(img_path)
    
    x_center = img.shape[0]/16 # Height of the center.
    y_center = img.shape[1]/16 # Width of the center.

    ltr_pos = "abcdefgh"
    chess_board_coords = {}
    for i in range(len(ltr_pos)):
        for j in range(1,9):
            curr_pos = ltr_pos[i]+str(j)
            curr_x = (2*i+1) * x_center
            curr_y = (2*j - 1) * y_center
            chess_board_coords[curr_pos] = [curr_x, curr_y]
    
    # Return the center coordinates of all 64 squares.
    return chess_board_coords


# -----------------------------------------------------------------------------------------------------------------
# Perform chess piece detection using YOLOv8 and get the best weights file ("best.pt")
# Then calculate the bounding boxes for all chess pieces and get bottom center's coordinates.
# -----------------------------------------------------------------------------------------------------------------

def find_Bottom_Centers(img_path):
    
    # Do detection using YOLOv8.
    model = YOLO('best.pt')
    results = model.predict(img_path, conf=0.40)

    board_img = cv2.imread(img_path)
    piece_bottom_centers = []

    for result in results:                                        
        boxes = result.boxes.cpu().numpy()                         
        for box in boxes:                                          
            r = box.xyxy[0].astype(int)   # Coordinates in the XYXY format.
            piece_bottom_centers.append([r[-1], (r[0]+r[2])/2, result.names[int(box.cls[0])]])                                                                       
            cv2.rectangle(board_img, r[:2], r[2:], (255, 255, 255), 2)   
    
    # Return the coordinates of all the chess piece's bottom centers.
    return piece_bottom_centers


# -----------------------------------------------------------------------------------------------------------------
# Find the positions on the chessboard using Approach 1 (see README.md).
# -----------------------------------------------------------------------------------------------------------------

def find_Positions_from_Labelled_Matrix(img_path, projective_matrix, piece_bottom_centers):
    
    img = cv2.imread(img_path)
    top_left = (0, 0)
    top_right = (img.shape[0], 0)
    bottom_left = (0, img.shape[1])
    bottom_right = (img.shape[0], img.shape[1])

    image_width = img.shape[0]
    image_height = img.shape[1]

    # Declare matrix of size image_width x image_height.
    matrix = np.zeros((image_width, image_height))
    
    # Get the cell height and width.
    cell_height = image_height/8
    cell_width = image_width/8

    # Generate the Labelled matrix.
    count = 1
    for i in range(0, 8):
        for j in range(0, 8):
            top_left = (int(i*cell_width), int(j*cell_height))
            bottom_right = (int((i+1)*cell_width), int((j+1)*cell_height))
            for i1 in range(top_left[0], bottom_right[0]):
                for j1 in range(top_left[1], bottom_right[1]):
                    matrix[i1,j1] = count
            count +=1

    dict_for_chessboard = {1:'a1', 2:'a2', 3:'a3', 4:'a4', 5:'a5', 6:'a6', 7:'a7', 8:'a8', 
                            9:'b1', 10:'b2', 11:'b3', 12:'b4', 13:'b5', 14:'b6', 15:'b7', 16:'b8',
                            17:'c1', 18:'c2', 19:'c3', 20:'c4', 21:'c5', 22:'c6', 23:'c7', 24:'c8',
                            25:'d1', 26:'d2', 27:'d3', 28:'d4', 29:'d5', 30:'d6', 31:'d7', 32:'d8',
                            33:'e1', 34:'e2', 35:'e3', 36:'e4', 37:'e5', 38:'e6', 39:'e7', 40:'e8',
                            41:'f1', 42:'f2', 43:'f3', 44:'f4', 45:'f5', 46:'f6', 47:'f7', 48:'f8',
                            49:'g1', 50:'g2', 51:'g3', 52:'g4', 53:'g5', 54:'g6', 55:'g7', 56:'g8',
                            57:'h1', 58:'h2', 59:'h3', 60:'h4', 61:'h5', 62:'h6', 63:'h7', 64:'h8'}

    final_pos = {}

    # Assign the positions after fidning the label of the transformed point.
    for piece_bottom in piece_bottom_centers:
        piece_position = ""
        chess_piece = piece_bottom[-1]
        original_point = np.array([[piece_bottom[1], piece_bottom[0]-10]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(original_point, projective_matrix).astype(int)
        piece_position = dict_for_chessboard[matrix[transformed_point[0][0][1], transformed_point[0][0][0]]]
        
        if chess_piece not in final_pos.keys():
            final_pos[chess_piece] = [piece_position]
        else:
            final_pos[chess_piece].append(piece_position)

    # Return the position dictionary.
    return final_pos


# -----------------------------------------------------------------------------------------------------------------
# Find the positions on the chessboard using Approach 2 (see README.md).
# -----------------------------------------------------------------------------------------------------------------

def find_Positions_from_Board_Centers(projective_matrix, chess_board_coords, piece_bottom_centers):
    
    final_pos = {}

    # Find the minimum distance B/W the chess piece and all the 64 squares.
    # Use L2 norm as the distance metric.

    for piece_bottom in piece_bottom_centers:
        min_dist_from_center = float('inf')
        piece_position = ""
        chess_piece = piece_bottom[-1]

        original_point = np.array([[piece_bottom[1], piece_bottom[0]-5]], dtype=np.float32).reshape(-1, 1, 2)
        for chess_pos in chess_board_coords.keys():
            transformed_point = cv2.perspectiveTransform(original_point, projective_matrix)
            x2, y2 = transformed_point[0][0][1], transformed_point[0][0][0]
            x1, y1 = chess_board_coords[chess_pos][0], chess_board_coords[chess_pos][1]
            d = math.dist([x1,y1],[x2,y2])
            if(min_dist_from_center > d):
                min_dist_from_center = d
                piece_position = chess_pos
        
        if chess_piece not in final_pos.keys():
            final_pos[chess_piece] = [piece_position]
        else:
            final_pos[chess_piece].append(piece_position)
    
    # Return the position dictionary.
    return final_pos


# -----------------------------------------------------------------------------------------------------------------
# Function to obtain the positions on the chess board. Use either approach 1 or approach 2.
# -----------------------------------------------------------------------------------------------------------------

def get_Positions(img_path):

    slant_img = find_Hough_Lines(img_path)
    x_coords, y_start_coords, y_end_coords = find_Square_Boundaries(slant_img)
    img_output, projective_matrix = find_Perspective_Transform(img_path, x_coords, y_start_coords, y_end_coords)
    chess_board_centers = find_ChessBoard_Centers(img_path)
    piece_bottom_centers = find_Bottom_Centers(img_path)
    # final_pos = find_Positions_from_Board_Centers(projective_matrix, chess_board_centers, piece_bottom_centers)
    final_pos = find_Positions_from_Labelled_Matrix(img_path, projective_matrix, piece_bottom_centers)

    return final_pos


# -----------------------------------------------------------------------------------------------------------------
# Find the Forsyth - Edwards Notation (FEN) from the positions dictionary.
# -----------------------------------------------------------------------------------------------------------------

def get_fen(final_pos, move):
    
    # Annotate the pieces.
    piece_annotations = {'white-rook':'R', 'white-knight':'N', 'white-bishop':'B', 'white-queen':'Q', 'white-king':'K', 'white-pawn':'P',
                     'black-rook':'r', 'black-knight':'n', 'black-bishop':'b', 'black-queen':'q', 'black-king':'k', 'black-pawn':'p'}
    
    # Generate the position matrix.
    pos_mat = [[' ' for i in range(8)] for j in range(8)]

    for chess_piece in final_pos.keys():
        for pos in final_pos[chess_piece]:
            i = ord(pos[0]) - 96
            j = int(pos[1])
            pos_mat[i-1][j-1] = piece_annotations[chess_piece]
    
    FEN = ""

    # Rearrange the position matrix to align with the StockFish Notation.
    
    pos_mat = np.array(pos_mat).transpose() # Take transpose.
    pos_mat = np.flipud(pos_mat) # Flip it upside down.
    
    # Parse all the 64 squares and generate the FEN.
    for i in range(8):
        row_fen = ""
        empty_count = 0
        for j in range(8):
            if(pos_mat[i][j] != ' '):
                if(empty_count == 0):
                    row_fen += pos_mat[i][j]
                else:
                    row_fen += str(empty_count) + pos_mat[i][j]
                    empty_count = 0

            else:
                empty_count += 1
        
        FEN += row_fen 
        if(empty_count != 0):
            FEN += str(empty_count)
        FEN += '/'
    
    move_number = random.randint(0,6)
    move_id = random.randint(6, 10)
    FEN = FEN[:-1] + " " + move +  " - - " + str(move_number) + ' ' + str(move_id)
    
    # Return the FEN representation.
    return FEN
