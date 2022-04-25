import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        (i, j) = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return self.cells
        else:
            return set()


    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return set()
        

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count - 1
        

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)

        


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        
        
        #marks cell as move made
        self.moves_made.add(cell)
        
        #adds cell to safe
        self.safes.add(cell)


        new_sent = Sentence(list(self.neighbors(cell)), count)
        print("new sentence:", new_sent)


        #marks already known mines & safe
        for mine in self.mines:
            new_sent.mark_mine(mine)

        for safe in self.safes:
            new_sent.mark_safe(safe)

        new_sent = Sentence(set(new_sent.cells), new_sent.count)
        print("marked sentence", new_sent)
        
        if new_sent not in self.knowledge:
            self.knowledge.append(new_sent)

        
        #makes new inferences
        for sentence in self.knowledge:
            
            #convert all sentences to list for removal
            sentence.cells = list(sentence.cells)
            
            for mine in sentence.known_mines():
                print("mine found", mine)
                if mine not in self.mines:
                    self.mines.add(mine)
                    
                if mine in sentence.cells:
                    print("marking mine...", mine)
                    sentence.mark_mine(mine)
                    

            for safe in sentence.known_safes():                
                if safe not in self.safes:
                    self.safes.add(safe)
                    
                if safe in sentence.cells:
                    print("marking safes...", safe)
                    sentence.mark_safe(safe)
        
        
            sentence.cells = set(sentence.cells)

            if sentence == Sentence(set(), 0) or sentence == Sentence([], 0):
                print("removing sentence...", sentence)
                self.knowledge.remove(sentence)
                         
                
        #makes inferences using the subset method
        for sent1, sent2 in list(itertools.combinations(self.knowledge, 2)):
            sent1.cells = set(sent1.cells)
            sent2.cells = set(sent2.cells)
            print("sentence 1 & 2", sent1, sent2)
            
            
            if sent1.cells.issubset(sent2.cells):
                    
                new_sent = Sentence((sent2.cells - sent1.cells), sent2.count - sent1.count)
            elif sent2.cells.issubset(sent1.cells):
                new_sent = Sentence((sent1.cells - sent2.cells), sent1.count - sent2.count)
    
            if new_sent not in self.knowledge:
                print("appending new sentence...", new_sent)
                self.knowledge.append(new_sent)

        
        
            

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.
        """
        safe_moves = list(self.safes - self.moves_made)

        if len(safe_moves) > 1:
            return safe_moves[random.randint(0, len(safe_moves)-1)]
        elif len(safe_moves) == 1:
            return safe_moves[0]
        else:
            return None


        

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if self.make_safe_move() == None:
            i = random.randint(0, self.height - 1)
            j = random.randint(0, self.width - 1)

            if (i, j) not in (self.moves_made and self.mines):
                return (i,j)
            else:
                return self.make_random_move()


    def neighbors(self, cell):

        nbs = set()
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):

                if (i, j) == cell:
                    pass

                elif 0 <= i < self.height and 0 <= j < self.width:
                    nbs.add((i,j))

        return nbs

board = MinesweeperAI()