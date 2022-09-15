#!/usr/bin/env python3
import numpy as np

def possible(grid, x, y, n):
    for i in range(9):
        if grid[x][i] == n:
            return False
    for j in range(9):
        if grid[j][y] == n:
            return False
    
    box_x = (x // 3) * 3
    box_y = (y // 3) * 3

    for i in range(9):
        for j in range(9):
            if grid[box_x + i][box_y + j] == n:
                return False
    return True

def solve(grid):
    for x in range(9):
        for y in range(9):
            if grid[x][y] == 0:
                for n in range(1, 10):
                    if possible(grid, x, y, n):
                        grid[x][y] = n
                        solve()
                        grid[x][y] = 0
                return
