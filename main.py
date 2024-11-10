import cv2
import pyautogui
import numpy as np
import math
from sympy import Matrix
from time import sleep


LIGHT_GREEN = (170, 215, 81)
GREEN = (162, 209, 73)
DARK_GREEN = (135, 175, 58)
LIGHT_BROWN = (229, 194, 159)
DARK_BROWN = (215, 184, 153)
RED = (230, 51, 7)

class Board:
    def __init__(self):
        self.top_left = None
        self.bottom_right = None
        self.pixel_size = None
        self.game_started = True
        self.board_size = self.get_board_bounds()
        self.board = self.get_board()
        self.matrix, self.unknown_tiles = self.board_to_matrix()
        self.next_moves, self.to_flag = self.find_next_moves()
        
    # sets the board bounds (x, y), returns board size (x, y)
    def get_board_bounds(self):
        def find_nth_match(image, color, n=0):
            match = np.where(np.all(np.abs(image - color) <= 0, axis=-1))
            if len(match[0]) == 0: return (0, 0)
            return (match[1][n], match[0][n])
        
        def find_corner(coords, side):
            corner = coords[0]
            for coord in coords:
                if coord == (0, 0): continue
                if side == 'left':
                    if coord[0] + coord[1] < corner[0] + corner[1]:
                        corner = coord
                elif side == 'right':
                    if coord[0] + coord[1] > corner[0] + corner[1]:
                        corner = coord
            return corner
        # sets board bounds, returns true if board is found
        def get_bounds():
            screenshot = pyautogui.screenshot()
            image = np.array(screenshot)

            first_light_green = find_nth_match(image, LIGHT_GREEN)
            first_light_brown = find_nth_match(image, LIGHT_BROWN)
            first_green = find_nth_match(image, GREEN)
            first_dark_brown = find_nth_match(image, DARK_BROWN)
            first_dark_green = find_nth_match(image, DARK_GREEN)
            last_light_green = find_nth_match(image, LIGHT_GREEN, -1)
            last_light_brown = find_nth_match(image, LIGHT_BROWN, -1)
            
            if first_dark_green == (0, 0): self.game_started = False 
            
            self.top_left = find_corner([first_light_green, first_light_brown], 'left')
            top_left_dark = find_corner([first_green, first_dark_green, first_dark_brown], 'left')
            self.bottom_right = find_corner([last_light_green, last_light_brown], 'right')
            
            self.pixel_size = top_left_dark[0] - self.top_left[0]
            
            return True
        
        if not get_bounds():
            return None
            
        board_size = math.ceil((self.bottom_right[0] - self.top_left[0]) / self.pixel_size), \
                     math.ceil((self.bottom_right[1] - self.top_left[1]) / self.pixel_size)
        return board_size
    
    # returns list of positions where template is found in image
    def search_image(self, image, template):
        #img = cv2.imread(image)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        template_img = cv2.imread(template, 0)

        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        positions = []
        for pt in zip(*loc[::-1]):
            positions.append(pt)

        return positions
    
    # converts pixel coordinates to board indices
    def coordinates_to_index(self, x, y):
        return (x - self.top_left[0]) / self.pixel_size, (y - self.top_left[1]) / self.pixel_size

    # creates board from image
    def get_board(self):
        screenshot = pyautogui.screenshot()
        npimage = np.array(screenshot)
        cv2image = cv2.cvtColor(npimage, cv2.COLOR_RGB2BGR)
        # have to test which images to match since theres two sets for some reason, have to add take screenshot
        # can speed this up by making it a runtine length change
        def get_tile_images(game_image):
            test_1 = len(self.search_image(game_image, 'images/one.png'))
            test_2 = len(self.search_image(game_image, 'images/1.png'))
            if test_1 > test_2:
                return ['images/one.png', 'images/two.png', 'images/three.png', 'images/four.png', 'images/five.png', 'images/six.png']
            else:
                return ['images/1.png', 'images/2.png', 'images/3.png', 'images/4.png', 'images/5.png', 'images/6.png']

        tile_images = get_tile_images(cv2image)

        # set up board with 0 for clicked tiles, -1 for unknown tiles
        board = []
        for y in range(self.top_left[1] + 5, self.bottom_right[1], self.pixel_size):
            row = []
            for x in range(self.top_left[0] + 5, self.bottom_right[0], self.pixel_size):
                color = (cv2image[y][x][2], cv2image[y][x][1], cv2image[y][x][0]) # BGR --> RGB
                tile = 0 if (color == LIGHT_BROWN) or (color == DARK_BROWN) else -1
                row.append(tile) 
            board.append(row)

        # fill in known tiles with numbers
        for k in range(len(tile_images)):
            positions = self.search_image(cv2image, tile_images[k])
            for pos in positions:
                x, y = pos
                i, j = self.coordinates_to_index(x, y)
                board[int(j)][int(i)] = k + 1

        return board

    # searches around a tile for an item. returns all positions of item (x, y)
    def search_nearby(self, board, tile, item):
        positions = []
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0: continue
                if tile[0] + x < 0 or tile[0] + x >= self.board_size[0]: continue
                if tile[1] + y < 0 or tile[1] + y >= self.board_size[1]: continue

                if board[tile[1] + y][tile[0] + x] == item:
                    positions.append((tile[0] + x, tile[1] + y))
        return positions
        
    # bad code refactor?
    # converts board to matrix, returns matrix and list of unknown tiles
    def board_to_matrix(self):
        if not self.game_started:
            return [], []
        # put all edge "?" on a list x = [(x,y), (x,y)...]
        # for loop every number
        # add number to augmented col
        # if number has ? around it find indexof ? on the list and 1 same spot
        matrix = []
        unknown_tiles = []
        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                if self.board[y][x] > 0:
                    surrounding_unknown = self.search_nearby(self.board, (x, y), -1)
                    unknown_tiles += [coord for coord in surrounding_unknown if coord not in unknown_tiles]

                    row = [0] * len(unknown_tiles)
                    for coord in surrounding_unknown:
                        index = unknown_tiles.index(coord)
                        row[index] = 1
                    row.append(self.board[y][x])
                    matrix.append(row)

        for row in matrix:
            row[-1:-1] = [0] * (len(unknown_tiles) - len(row) + 1)
        return matrix, unknown_tiles
    
    def is_flagged(self, coord):
        coord = self.index_to_coordinates(coord, 7)
        screenshot = pyautogui.screenshot()
        npimage = np.array(screenshot)
        color = npimage[coord[1], coord[0]]
        return color[0] == RED[0] and color[1] == RED[1] and color[2] == RED[2]

    # returns next moves and tiles to flag (y, x)
    def find_next_moves(self):
        if not self.game_started:
            return set(), set()
        
        next_moves = set()
        next_to_flag = set()
        matrix = Matrix(self.matrix)
        rref_matrix, pivot_cols = matrix.rref()
        rref_matrix = rref_matrix.tolist() # convert symPy array to list
        for row in rref_matrix: # add cases 1 -1 1, 1 -1 -1
            #print(row)
            if row[:-1].count(1) == row[-1]:
                for i in range(len(row) - 1):
                    if row[i] == 1:
                        coord = self.unknown_tiles[i]
                        if not self.is_flagged(coord):
                            self.board[coord[1]][coord[0]] = -2
                            next_to_flag.add((coord))
            if row.count(0) == len(row) - 1:
                #print(row)
                next_move_index = row.index(1)
                x, y = self.unknown_tiles[next_move_index]
                next_moves.add((x, y))
            
            if row[:-1].count(-1) == 1 and row[:-1].count(1) == 1:
                if row[-1] == 1:
                    next_move_index = row.index(-1)
                    to_flag_index = row.index(1)
                elif row[-1] == -1:
                    next_move_index = row.index(1)
                    to_flag_index = row.index(-1)
                else:
                    continue
                move_coord = self.unknown_tiles[next_move_index]
                flag_coord = self.unknown_tiles[to_flag_index]
                next_moves.add(move_coord)
                if not self.is_flagged(flag_coord):# may cause bug where player flags but solver doesnt know
                    self.board[flag_coord[1]][flag_coord[0]] = -2 
                    next_to_flag.add(flag_coord)
            

        return next_moves, next_to_flag
    
    def find_mineable(self):
        # search if a number has an empty square around it, if yes, 
        # search if the number of flags around it is equal to the number on the tile
        # if yes, click all empty squares around it/chord it

        # have to update board after mining tiles
        self.board = self.get_board()
        for coord in self.to_flag:
            self.board[coord[1]][coord[0]] = -2

        mineable = set()
        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                if self.board[y][x] > 0:
                    surrounding_unknown = self.search_nearby(self.board, (x, y), -1)
                    if len(surrounding_unknown) == 0: continue
                    surrounding_flags = self.search_nearby(self.board, (x, y), -2)
                    if len(surrounding_flags) != self.board[y][x]: continue
                    for coord in surrounding_unknown:
                        mineable.add((coord[0], coord[1]))
        return mineable

    def check_if_solved(self):
        pass


    
    # converts board indices to pixel coordinates
    # offset to click in the middle of the tile
    def index_to_coordinates(self, coord, offset=5):
        return (self.top_left[0] + coord[0] * self.pixel_size + offset, self.top_left[1] + coord[1] * self.pixel_size + offset)

    def click_tile(self, coord, button):
        coord = self.index_to_coordinates(coord)
        pyautogui.click(x=coord[0], y=coord[1], button=button)

    def run_solver(self):
        if not self.game_started:
            middle = (self.board_size[0] // 2, self.board_size[1] // 2)
            self.click_tile(middle, 'left')

        while len(self.next_moves) > 0 or not self.game_started:
            self.game_started = True
            #for y, x in self.to_flag:
            #    self.click_tile(x, y, 'right')
            for coord in self.next_moves:
                self.click_tile(coord, 'left')
            sleep(0.3)
            mineable = self.find_mineable()
            for coord in mineable:
                self.click_tile(coord, 'left')

            #if self.check_if_solved():
            #    break
            self.board = self.get_board()
            self.matrix, self.unknown_tiles = self.board_to_matrix()
            self.next_moves, self.to_flag = self.find_next_moves()


        



        
game = Board()

game.run_solver()
# board: 0-6: #num mines, -1: unknown, -2: flagged
# TODO: get better 6,7,8 images (might not work with current images)

# make it faster by not mining mineable tiles but finding the chord move
# add probability to the solver (if no definite moves, make a guess) have to resaeach matrices for this
# MAJOR BUG OFFSET IS HAVING MIND THAT ITS MEDIUM DIFF