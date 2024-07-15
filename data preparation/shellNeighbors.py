import chimera
import numpy
from numpy import *


#==============================================================================
# Shell CLASS
#==============================================================================

class Shell(object):
    def __init__(self, atoms_L, minR, maxR):
        self.minR = minR
        self.maxR = maxR
        self.atoms_L = atoms_L
        self.gridDict = {}
        self.minVals = self.__minValsOfCoords()

        # To answer near neighbour queries we create a data structure that is a dictionary
        # containing lists of atoms taken from the atmList.
        # A dictionary key will be a list of three integers.  The three integers are the
        # coordinates of a cube in a virtual 3D grid (virtual because we do not actually
        # construct such a data structure.
        # The value corresponding to a key is the list of atoms that end up in this cube of the
        # virtual 3D grid when it is superimposed over the collection of atoms in atoms_L.

        for atm in self.atoms_L:
            # Generate integer coordinates for the cube containing this atom.
            gridCoords = self.__convCoordsToGridCoords(numpy.array(atm.coord()))
            if not self.gridDict.has_key(gridCoords):
                self.gridDict[gridCoords] = []
            self.gridDict[gridCoords].append(atm)


    # -----------------------------------------------------------------------------
    # Function to compute the three minimum values of the coordinate entries.
    def __minValsOfCoords(self):
        xMin = yMin = zMin =  99999.0    # Initialize to high values.
        for atm in self.atoms_L:
            cc = atm.coord()
            if cc[0] < xMin: xMin = cc[0]
            if cc[1] < yMin: yMin = cc[1]
            if cc[2] < zMin: zMin = cc[2]
        return [xMin, yMin, zMin]


    # -----------------------------------------------------------------------------
    # Function to convert 3D coordinates into grid coordinates.
    def __convCoordsToGridCoords(self, coords):
        fV = floor((numpy.array(coords) - self.minVals)/self.maxR)
        return (int(fV[0]), int(fV[1]), int(fV[2]))
        
    # -----------------------------------------------------------------------------
    def getAtomsInShell(self, centerAtom):
        # This function accepts an atom and returns a list
        # of all atoms that are within the shell surrounding the centerAtom.
        # The is specified by minimum radius minR and maximum radius maxR.

        atomsInShell_L = []
        
        # Generate integer coordinates for the cube containing the specified atom.
        gridCoords = self.__convCoordsToGridCoords(centerAtom.coord())
        xI = gridCoords[0]
        yI = gridCoords[1]
        zI = gridCoords[2]
        
        # Now inspect the 27 cubes and get the atoms that are within maxR of the centerAtom.
        for i in range(xI - 1, xI + 2):
            for j in range(yI - 1, yI +2):
                for k in range(zI - 1, zI + 2):
                    coordTuple = (i, j, k)
                    if self.gridDict.has_key(coordTuple):
                        nbrList = self.gridDict[coordTuple]
                        for atm in nbrList:
                            interAtomDistance = numpy.linalg.norm(centerAtom.coord() - atm.coord())
                            if interAtomDistance <= self.maxR:
                                # Do not include the atom specified by atm_Coords!
                                if interAtomDistance < 0.01: continue
                                # Eliminate atoms closer than the minimum radius.
                                if interAtomDistance < self.minR: continue 
                                atomsInShell_L.append(atm)

        return atomsInShell_L






