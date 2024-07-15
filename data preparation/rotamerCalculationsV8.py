import chimera, shellNeighbors, sets
from chimera import runCommand, Element
from chimera.molEdit import addAtom, addBond
from shellNeighbors import Shell
from sets import Set
import Rotamers
from Rotamers import getRotamers, useRotamer, NoResidueRotamersError
import numpy
from numpy import array, linalg
import operator
from operator import itemgetter
from MMMD import base


#==============================================================================
def validResidue(r):
    if not r.type in ('CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE',
                      'PRO', 'THR', 'PHE', 'ASN', 'MET', 'HIS',
                      'LEU', 'ARG', 'TRP', 'VAL', 'GLU', 'TYR'): return False
    if not r.findAtom("N"): return False
    if not r.findAtom("CA"): return False    
    if not r.findAtom("C"): return False
    return True

#==============================================================================
# Lennard Jones evaluation CLASS
#==============================================================================

class LJ_evaluation(object):
    def __init__(self):
        # Map from residue and atom names to Amber Atom Types:
        self.elementNameMap_D = {':ALA@C': 'C',    ':ALA@CA': 'CT',  ':ALA@CB': 'CT',  ':ALA@H': 'H', ':ALA@HA': 'H1',
                                 ':ALA@HB1': 'HC', ':ALA@HB2': 'HC', ':ALA@HB3': 'HC', ':ALA@N': 'N', ':ALA@O': 'O', ':ALA@OXT': 'O',
                                 ':ALA@H1': 'H1', ':ALA@H2': 'H1', ':ALA@H3': 'H1', 

                                 ':ARG@C': 'C', ':ARG@CA': 'CT', ':ARG@CB': 'CT', ':ARG@CD': 'CT', ':ARG@CG': 'CT',
                                 ':ARG@CZ': 'CA', ':ARG@H': 'H', ':ARG@HA': 'H1', ':ARG@HB2': 'HC', ':ARG@HB3': 'HC',
                                 ':ARG@HD2': 'H1', ':ARG@HD3': 'H1', ':ARG@HE': 'H', ':ARG@HG2': 'HC', ':ARG@HG3': 'HC',
                                 ':ARG@HH11': 'H', ':ARG@HH12': 'H', ':ARG@HH21': 'H', ':ARG@HH22': 'H', ':ARG@N': 'N',
                                 ':ARG@NE': 'N2', ':ARG@NH1': 'N2', ':ARG@NH2': 'N2', ':ARG@O': 'O', ':ARG@OXT': 'O',
                                 ':ARG@H1': 'H1', ':ARG@H2': 'H1', ':ARG@H3': 'H1',

                                 ':ASP@C': 'C', ':ASP@CA': 'CT', ':ASP@CB': 'CT', ':ASP@CG': 'C', ':ASP@H': 'H', ':ASP@HA': 'H1',
                                 ':ASP@HB2': 'HC', ':ASP@HB3': 'HC', ':ASP@N': 'N', ':ASP@O': 'O', ':ASP@OD1': 'O2', ':ASP@OD2': 'O2', ':ASP@OXT': 'O',
                                 ':ASP@H1': 'H1', ':ASP@H2': 'H1', ':ASP@H3': 'H1',

                                 ':ASN@C': 'C', ':ASN@CA': 'CT', ':ASN@CB': 'CT', ':ASN@CG': 'C', ':ASN@H': 'H', ':ASN@HA': 'H1',
                                 ':ASN@HB2': 'HC', ':ASN@HB3': 'HC', ':ASN@HD21': 'H', ':ASN@HD22': 'H', ':ASN@N': 'N',
                                 ':ASN@ND2': 'N', ':ASN@O': 'O', ':ASN@OD1': 'O', ':ASN@OXT': 'O', ':ASN@HN1': 'H', ':ASN@HN2': 'H',
                                 ':ASN@H1': 'H1', ':ASN@H2': 'H1', ':ASN@H3': 'H1', ':ASN@HC2': 'HC', ':ASN@HC3': 'HC', ':ASN@HC': 'H1',

                                 ':CYS@C': 'C', ':CYS@CA': 'CT', ':CYS@CB': 'CT', ':CYS@H': 'H', ':CYS@HA': 'H1', ':CYS@HB2': 'HC',
                                 ':CYS@HB3': 'HC', ':CYS@HG': 'HS', ':CYS@N': 'N', ':CYS@O': 'O', ':CYS@SG': 'SH', ':CYS@OXT': 'O',
                                 ':CYS@H1': 'H1', ':CYS@H2': 'H1', ':CYS@H3': 'H1', 

                                 ':GLU@C': 'C', ':GLU@CA': 'CT', ':GLU@CB': 'CT', ':GLU@CD': 'C', ':GLU@CG': 'CT',
                                 ':GLU@H': 'H', ':GLU@HA': 'H1', ':GLU@HB2': 'HC', ':GLU@HB3': 'HC', ':GLU@HG2': 'HC',
                                 ':GLU@HG3': 'HC', ':GLU@N': 'N', ':GLU@O': 'O', ':GLU@OE1': 'O2', ':GLU@OE2': 'O2', ':GLU@OXT': 'O',
                                 ':GLU@H1': 'H1', ':GLU@H2': 'H1', ':GLU@H3': 'H1', 

                                 ':GLN@C': 'C', ':GLN@CA': 'CT', ':GLN@CB': 'CT', ':GLN@CD': 'C', ':GLN@CG': 'CT', ':GLN@H': 'H',
                                 ':GLN@HA': 'H1', ':GLN@HB2': 'HC', ':GLN@HB3': 'HC', ':GLN@HE21': 'H', ':GLN@HE22': 'H', ':GLN@HG2': 'HC',
                                 ':GLN@HG3': 'HC', ':GLN@N': 'N', ':GLN@NE2': 'N', ':GLN@O': 'O', ':GLN@OE1': 'O', ':GLN@OXT': 'O',
                                 ':GLN@H1': 'H1', ':GLN@H2': 'H1', ':GLN@H3': 'H1', 

                                 ':GLY@C': 'C', ':GLY@CA': 'CT', ':GLY@H': 'H', ':GLY@HA2': 'H1', ':GLY@HA3': 'H1',
                                 ':GLY@N': 'N', ':GLY@O': 'O', ':GLY@OXT': 'O',
                                 ':GLY@H1': 'H1', ':GLY@H2': 'H1', ':GLY@H3': 'H1', 

                                 ':HIS@C': 'C', ':HIS@CA': 'CT', ':HIS@CB': 'CT', ':HIS@CD2': 'C', ':HIS@CE1': 'C', ':HIS@CG': 'C',
                                 ':HIS@H': 'H', ':HIS@HA': 'H1', ':HIS@HB2': 'HC', ':HIS@HB3': 'HC', ':HIS@HD2': 'H4', ':HIS@HE1': 'H5',
                                 ':HIS@HE2': 'H', ':HIS@N': 'N', ':HIS@ND1': 'N', ':HIS@NE2': 'N', ':HIS@O': 'O', ':HIS@OXT': 'O',
                                 ':HIS@H1': 'H1', ':HIS@H2': 'H1', ':HIS@H3': 'H1', ':HIS@HD1': 'H',

                                 ':ILE@C': 'C', ':ILE@CA': 'CT', ':ILE@CB': 'CT', ':ILE@CD1': 'CT', ':ILE@CG1': 'CT', ':ILE@CG2': 'CT',
                                 ':ILE@H': 'H', ':ILE@HA': 'H1', ':ILE@HB': 'HC', ':ILE@HD11': 'HC', ':ILE@HD12': 'HC', ':ILE@HD13': 'HC',
                                 ':ILE@HG12': 'HC', ':ILE@HG13': 'HC', ':ILE@HG21': 'HC', ':ILE@HG22': 'HC', ':ILE@HG23': 'HC',
                                 ':ILE@N': 'N', ':ILE@O': 'O', ':ILE@OXT': 'O',
                                 ':ILE@H1': 'H1', ':ILE@H2': 'H1', ':ILE@H3': 'H1', 

                                 ':LEU@C': 'C', ':LEU@CA': 'CT', ':LEU@CB': 'CT', ':LEU@CD1': 'CT', ':LEU@CD2': 'CT', ':LEU@CG': 'CT',
                                 ':LEU@H': 'H', ':LEU@HA': 'H1', ':LEU@HB2': 'HC', ':LEU@HB3': 'HC', ':LEU@HD11': 'HC', ':LEU@HD12': 'HC',
                                 ':LEU@HD13': 'HC', ':LEU@HD21': 'HC', ':LEU@HD22': 'HC', ':LEU@HD23': 'HC', ':LEU@HG': 'HC',
                                 ':LEU@N': 'N', ':LEU@O': 'O', ':LEU@OXT': 'O',
                                 ':LEU@H1': 'H1', ':LEU@H2': 'H1', ':LEU@H3': 'H1', 

                                 ':LYS@C': 'C', ':LYS@CA': 'CT', ':LYS@CB': 'CT', ':LYS@CD': 'CT', ':LYS@CE': 'CT', ':LYS@CG': 'CT',
                                 ':LYS@H': 'H', ':LYS@HA': 'H1', ':LYS@HB2': 'HC', ':LYS@HB3': 'HC', ':LYS@HD2': 'HC', ':LYS@HD3': 'HC',
                                 ':LYS@HE2': 'HP', ':LYS@HE3': 'HP', ':LYS@HG2': 'HC', ':LYS@HG3': 'HC', ':LYS@HZ1': 'H',
                                 ':LYS@HZ2': 'H', ':LYS@HZ3': 'H', ':LYS@N': 'N', ':LYS@NZ': 'N3', ':LYS@O': 'O', ':LYS@OXT': 'O',
                                 ':LYS@H1': 'H1', ':LYS@H2': 'H1', ':LYS@H3': 'H1', 

                                 ':MET@C': 'C', ':MET@CA': 'CT', ':MET@CB': 'CT', ':MET@CE': 'CT', ':MET@CG': 'CT', ':MET@H': 'H',
                                 ':MET@HA': 'H1', ':MET@HB2': 'HC', ':MET@HB3': 'HC', ':MET@HE1': 'H1', ':MET@HE2': 'H1', ':MET@HE3': 'H1',
                                 ':MET@HG2': 'H1', ':MET@HG3': 'H1', ':MET@N': 'N', ':MET@O': 'O', ':MET@SD': 'S', ':MET@OXT': 'O',
                                 ':MET@H1': 'H1', ':MET@H2': 'H1', ':MET@H3': 'H1', 

                                 ':PHE@C': 'C', ':PHE@CA': 'CT', ':PHE@CB': 'CT', ':PHE@CD1': 'CA', ':PHE@CD2': 'CA', ':PHE@CE1': 'CA',
                                 ':PHE@CE2': 'CA', ':PHE@CG': 'CA', ':PHE@CZ': 'CA', ':PHE@H': 'H', ':PHE@HA': 'H1', ':PHE@HB2': 'HC',
                                 ':PHE@HB3': 'HC', ':PHE@HD1': 'HA', ':PHE@HD2': 'HA', ':PHE@HE1': 'HA', ':PHE@HE2': 'HA',
                                 ':PHE@HZ': 'HA', ':PHE@N': 'N', ':PHE@O': 'O', ':PHE@OXT': 'O',
                                 ':PHE@H1': 'H1', ':PHE@H2': 'H1', ':PHE@H3': 'H1', 

                                 ':PRO@C': 'C', ':PRO@CA': 'CT', ':PRO@CB': 'CT', ':PRO@CD': 'CT', ':PRO@CG': 'CT', ':PRO@HA': 'H1',
                                 ':PRO@HB2': 'HC', ':PRO@HB3': 'HC', ':PRO@HD1': 'H1', ':PRO@HD2': 'H1', ':PRO@HD3': 'H1', ':PRO@HG2': 'HC',
                                 ':PRO@HG3': 'HC', ':PRO@N': 'N', ':PRO@O': 'O', ':PRO@OXT': 'O',
                                 ':PRO@H1': 'H1', ':PRO@H2': 'H1', ':PRO@H3': 'H1', 

                                 ':SER@C': 'C', ':SER@CA': 'CT', ':SER@CB': 'CT', ':SER@H': 'H', ':SER@HN1': 'H', ':SER@HN2': 'H', ':SER@HA': 'H1', ':SER@HB2': 'H1',
                                 ':SER@HB3': 'H1', ':SER@HG': 'HO', ':SER@N': 'N', ':SER@O': 'O', ':SER@OG': 'OH', ':SER@OXT': 'O',
                                 ':SER@H1': 'H1', ':SER@H2': 'H1', ':SER@H3': 'H1', ':SER@HC': 'H1', ':SER@HC2': 'H1', ':SER@HC3': 'H1', ':SER@HO': 'HO',

                                 ':THR@C': 'C', ':THR@CA': 'CT', ':THR@CB': 'CT', ':THR@CG2': 'CT', ':THR@H': 'H', ':THR@HA': 'H1',
                                 ':THR@HB': 'H1', ':THR@HG1': 'HO', ':THR@HG21': 'HC', ':THR@HG22': 'HC', ':THR@HG23': 'HC',
                                 ':THR@N': 'N', ':THR@O': 'O', ':THR@OG1': 'OH', ':THR@OXT': 'O',
                                 ':THR@H1': 'H1', ':THR@H2': 'H1', ':THR@H3': 'H1', 

                                 ':TRP@C': 'C', ':TRP@CA': 'CT', ':TRP@CB': 'CT', ':TRP@CD1': 'C', ':TRP@CD2': 'C', ':TRP@CE2': 'C',
                                 ':TRP@CE3': 'CA', ':TRP@CG': 'C', ':TRP@CH2': 'CA', ':TRP@CZ2': 'CA', ':TRP@CZ3': 'CA', ':TRP@H': 'H',
                                 ':TRP@HA': 'H1', ':TRP@HB2': 'HC', ':TRP@HB3': 'HC', ':TRP@HD1': 'H4', ':TRP@HE1': 'H', ':TRP@HE3': 'HA',
                                 ':TRP@HH2': 'HA', ':TRP@HZ2': 'HA', ':TRP@HZ3': 'HA', ':TRP@N': 'N', ':TRP@NE1': 'N', ':TRP@O': 'O', ':TRP@OXT': 'O',
                                 ':TRP@H1': 'H1', ':TRP@H2': 'H1', ':TRP@H3': 'H1', 

                                 ':TYR@C': 'C', ':TYR@CA': 'CT', ':TYR@CB': 'CT', ':TYR@CD1': 'CA', ':TYR@CD2': 'CA', ':TYR@CE1': 'CA',
                                 ':TYR@CE2': 'CA', ':TYR@CG': 'CA', ':TYR@CZ': 'C', ':TYR@H': 'H', ':TYR@HA': 'H1', ':TYR@HB2': 'HC',
                                 ':TYR@HB3': 'HC', ':TYR@HD1': 'HA', ':TYR@HD2': 'HA', ':TYR@HE1': 'HA', ':TYR@HE2': 'HA',
                                 ':TYR@HH': 'HO', ':TYR@N': 'N', ':TYR@O': 'O', ':TYR@OH': 'OH', ':TYR@OXT': 'O',
                                 ':TYR@H1': 'H1', ':TYR@H2': 'H1', ':TYR@H3': 'H1', 

                                 ':VAL@C': 'C', ':VAL@CA': 'CT', ':VAL@CB': 'CT', ':VAL@CG1': 'CT', ':VAL@CG2': 'CT', ':VAL@H': 'H',
                                 ':VAL@HA': 'H1', ':VAL@HB': 'HC', ':VAL@HG11': 'HC', ':VAL@HG12': 'HC', ':VAL@HG13': 'HC', ':VAL@HG21': 'HC',
                                 ':VAL@HG22': 'HC', ':VAL@HG23': 'HC', ':VAL@N': 'N', ':VAL@O': 'O', ':VAL@OXT': 'O',
                                 ':VAL@H1': 'H1', ':VAL@H2': 'H1', ':VAL@H3': 'H1'}

        
        self.radiusMap_D = {'H': 0.6000, 'HO': 0.0000, 'HS': 0.6000, 'HC': 1.4870, 'H1': 1.3870, 'H2': 1.2870, 'H3': 1.1870, 'HP': 1.1000,
                            'HA': 1.4590, 'H4': 1.4090, 'H5': 1.3590, 'HW': 0.0000, 'HZ': 1.487262,
                            'O': 1.6612, 'O2': 1.6612, 'OH': 1.7210, 'OS': 1.6837, 'OW': 1.778541, 'OZ': 1.778541,
                            'CT': 1.9080, 'CA': 1.9080, 'CM': 1.9080, 'C': 1.9080, 'CZ': 1.908185,
                            'N': 1.8240, 'N2': 1.8240, 'N3': 1.875,
                            'S': 2.0000, 'SH': 2.0000,
                            'P': 2.1000}

        self.epsMap_D = {'H': 0.0157, 'HO': 0.0000, 'HS': 0.0157, 'HC': 0.0157, 'H1': 0.0157, 'H2': 0.0157, 'H3': 0.0157, 'HP': 0.0157,
                         'HA': 0.0150, 'H4': 0.0150, 'H5': 0.0150, 'HW': 0.0000, 'HZ': 0.0157,
                         'O': 0.2100, 'O2': 0.2100, 'OH': 0.2104, 'OS': 0.1700, 'OW': 0.1553, 'OZ': 0.1553,
                         'CT': 0.1094, 'CA': 0.0860, 'CM': 0.0860, 'C': 0.0860, 'CZ': 0.1094,
                         'N': 0.1700, 'N2': 0.1700, 'N3': 0.1700,
                         'S': 0.2500, 'SH': 0.2500,
                         'P': 0.2000}

        self.maxExtensionMap_D = {'CYS': 3.06, 'ASP': 3.01, 'SER': 3.14, 'GLN': 4.55, 'LYS': 5.79, 'ILE': 3.44,
                                  'PRO': 3.32, 'THR': 3.24, 'PHE': 5.39, 'ASN': 3.37, 'MET': 5.01, 'HIS': 4.62,
                                  'LEU': 3.44, 'ARG': 6.96, 'TRP': 6.39, 'VAL': 2.98, 'GLU': 3.64, 'TYR': 6.06,
                                  'GLY': 0.00, 'ALA': 0.00}

        
        
        # Precompute a_ij and b_ij values:
        self.a_ij_D = {}
        self.b_ij_D = {}
        for aStr in self.epsMap_D.keys():
            for bStr in self.epsMap_D.keys():
                aRadius = self.radiusMap_D[aStr]
                if aStr[0] == 'H': aRadius = self.radiusMap_D[aStr]/2.0
                bRadius = self.radiusMap_D[bStr]
                if bStr[0] == 'H': bRadius = self.radiusMap_D[bStr]/2.0
                r_ijStar = aRadius + bRadius
                r_ijStar_2 = r_ijStar * r_ijStar
                r_ijStar_6 = r_ijStar_2 * r_ijStar_2 * r_ijStar_2
                r_ijStar_12 = r_ijStar_6 * r_ijStar_6
                eps_ij = numpy.sqrt(self.epsMap_D[aStr] * self.epsMap_D[bStr])
                self.a_ij_D[aStr + "#" + bStr] = eps_ij * r_ijStar_12
                self.b_ij_D[aStr + "#" + bStr] = 2.0 * eps_ij * r_ijStar_6


    def bondCount(self, a, b):
        oneBondAway_S = Set(a.neighbors)
        if b in oneBondAway_S: return 1
        twoBondsAway_S = Set()
        for aNbr in oneBondAway_S:
            twoBondsAway_S.union_update(Set(aNbr.neighbors))
        twoBondsAway_S.discard(a)
        if b in twoBondsAway_S: return 2
        threeBondsAway_S = Set()
        for aNbrNbr in twoBondsAway_S:
            threeBondsAway_S.union_update(Set(aNbrNbr.neighbors))
        if b in threeBondsAway_S: return 3
        return 4    # This value actually means 4 or more bonds.
        

    def atomPairEnergy(self, a, b):
        dist = linalg.norm(array(a.coord()) - array(b.coord()))
        if dist > 10.0: return 0.0  # Return 0 if interatomic distance > 10. (see Kingsford thesis).
        d_2 = dist * dist
        d_6 = d_2 * d_2 * d_2
        d_12 = d_6 * d_6
        aStr = self.elementNameMap_D[':' + a.residue.type + '@' + a.name]
        bStr = self.elementNameMap_D[':' + b.residue.type + '@' + b.name]
        pairEnergy = self.a_ij_D[aStr + "#" + bStr]/d_12 - self.b_ij_D[aStr + "#" + bStr]/d_6
        if abs(pairEnergy) < 0.000001: pairEnergy = 0.0
        if abs(a.residue.id.position - b.residue.id.position) > 1: return pairEnergy
        bC = self.bondCount(a,b)
        #print a.residue.type, a.residue.id.position, a.name, b.residue.type, b.residue.id.position, b.name, bC, pairEnergy  ################################33
        if bC <= 2: return 0.0
        if bC == 3: return min(pairEnergy/2.0, 100.0)  # Max allowed energy is 100 kcal/mol (see Kingsford thesis).
        return min(pairEnergy, 100.0)



    def residuePairEnergy(self, aRes, bRes):
        # This calculates the energy between two residues taking into account the side chain atoms only
        # Backbone atoms NOT included.
        aResBcarbon = aRes.findAtom('CB')
        if aRes.type == 'GLY': aResBcarbon = aRes.findAtom('CA')
        bResBcarbon = bRes.findAtom('CB')
        if bRes.type == 'GLY': bResBcarbon = bRes.findAtom('CA')
        bC_dist = linalg.norm(array(aResBcarbon.coord()) - array(bResBcarbon.coord()))
        if bC_dist > 8.0 + self.maxExtensionMap_D[aRes.type] + self.maxExtensionMap_D[bRes.type]: return 0.0
        
        energy = 0.0
        for a in aRes.atoms:
            if a.name in ('N', 'CA', 'C', 'O', 'HA', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'): continue
            for b in bRes.atoms:
                if b.name in ('N', 'CA', 'C', 'O', 'HA', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'): continue
                energy += self.atomPairEnergy(a, b)
        return energy

    def residueIntrinsicEnergy(self, aRes, statisticalEnergy, nearbyRes_L, fixedRes_L):
        # This calculates the sum of all atom pair energies such that one atom of a pair is in aRes (side chain only)
        # and the other atom of a pair is a backbone atom in a residue of the nearbyRes_L
        # OR the other atom is in a sidechain of a residue in the fixedRes_L.
        # It is to be expected that fixedRes_L is a subset of the nearbyRes_L
        # Typically, the fixed residues will be nearby and will be GLY, ALA or CYS (cysteine considered a fixed
        # residue if it is involved with a disulfide bridge to another CYS residue).
        energy = statisticalEnergy # This is the energy due to interactions between atoms within this sidechain.
        for a in aRes.atoms:
            if a.name in ('N', 'CA', 'C', 'O', 'HA', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'): continue # Skip backbone atoms of aRes.

            # Other atom is in the backbone of residues in nearbyRes_L:
            for nrbyR in nearbyRes_L:
                if abs(nrbyR.id.position - aRes.id.position) <= 1: continue # Skip nearby backbone (see pg. 32 of Kingsford thesis).
                for b in nrbyR.atoms:
                    if b.name in ('N', 'CA', 'C', 'O', 'HA', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'):
                        energy += self.atomPairEnergy(a, b)
                        
            # Other atom is in the sidechain of residues in fixedRes_L:
            for fixedR in fixedRes_L:
                for b in fixedR.atoms:
                    if b.name in ('N', 'CA', 'C', 'O', 'HA', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'): continue
                    energy += self.atomPairEnergy(a, b)

        return energy
            

#==============================================================================
# HydrogenAtom CLASS
#==============================================================================

class HydrogenAtom(object):
    # This is a "struct" to track the attributes of a hydrogen atom put in by addh.
    def __init__(self, hName, bondedTo, position):
        self.hName = hName
        self.bondedTo = bondedTo # Name of atom bonded to this hydrogen atom.
        self.position = position

#==============================================================================
# RotamerPlusHatoms CLASS
#==============================================================================

class RotamerPlusHatoms(object):
    # Tracks all the information related to a single rotamer at a particular residue site.
    def __init__(self, ix, siteResidue):
        self.rotamerIndex = ix  # Needed in some applications as a rotamer identifier.
        self.siteResidue = siteResidue
        self.rotamer = getRotamers(self.siteResidue)[1][self.rotamerIndex]
        self.hydrogens_L = []
        
    # -----------------------------------------------------------------------------
    # Function to get information about hydrogen atoms put in with the addh command.
    def generateHydrogenList(self):
        if self.siteResidue.type == "GLY" or self.siteResidue.type == "ALA": return
        useRotamer(self.siteResidue, [self.rotamer])
        runCommand("addh")
        for aa in self.siteResidue.atoms:
            if aa.name == "H": continue
            if aa.name[0] == 'H':
                    self.hydrogens_L.append(HydrogenAtom(aa.name, aa.neighbors[0].name, aa.coord()))
                                            
        for entry in self.hydrogens_L: ####################################################################### DEBUG
            print entry.hName, entry.bondedTo, entry.position ################################################ DEBUG

    # -----------------------------------------------------------------------------
    # Function to decorate a new rotamer with its hydrogen atoms.
    # To be called when a new rotamer is being used and after the generateHydrogenList has been called.
    def putInHydrogens(self):
        for h_Atom in self.hydrogens_L:
            nbr = self.siteResidue.findAtom(h_Atom.bondedTo)
            addBond(addAtom(h_Atom.hName, Element('H'), self.siteResidue, h_Atom.position), nbr)

                    
#==============================================================================
# SingleRotamer CLASS
#==============================================================================

class SingleRotamer(RotamerPlusHatoms):
    # Tracks all the information related to a single rotamer at a particular residue site.
    def __init__(self, ix, siteResidue, statisticalSelfEnergy):
        RotamerPlusHatoms.__init__(self, ix, siteResidue)
        self.statisticalSelfEnergy = statisticalSelfEnergy
        
        RotamerPlusHatoms.generateHydrogenList(self)
        # The next statement assumes that the previous statement has set up the residue site
        # with this rotamer and the hydrogen atoms are in place.
        self.intrinsicEnergy = lj.residueIntrinsicEnergy(self.siteResidue, self.statisticalSelfEnergy, nbrsMap_D_G[self.siteResidue], fixedRes_L_G)

        # The following are subject to change as dead end elimination progresses:
        self.sumMinPairEnergies = 0.0
        self.sumMaxPairEnergies = 0.0
        print "Building rotamer object. ResType: ", self.siteResidue.type, "resIndex: ", self.siteResidue.id.position, "rotNumber: ", self.rotamerIndex, "Intrinsic energy: ", self.intrinsicEnergy

    def updateRotPairEnergyDictionary(self):
        global allRotamers_G, lineCount_G, rotamerPairEnergy_D_G
        # This can only be called after all the Rotamer objects have been instantiated.
        # Compute the energy interactions between this rotamer and all the rotamers that
        # are at positions corresponding to nearby residues. These values to be stored in
        # the global dictionary rotamerPairEnergy_D_G.  Several local dictionaries were shown to be too slow.
        # Key is a tuple: (residue position, rotamer number, residue position, rotamer number).
        # Value is the corresponding pair energy.
        #
        # Skip if this ResRots object corresponds to a glycine or an alanine.
        if self.siteResidue.type == "ALA" or self.siteResidue.type == "GLY": return
        
        useRotamer(self.siteResidue, [self.rotamer])
        RotamerPlusHatoms.putInHydrogens(self)
        for resNbr in nbrsMap_D_G[self.siteResidue]:
            if not resNbr in residuesWithRotamers_L_G: continue
            # Do calculation only for neighbouring residues that are further along in position:
            if resNbr.id.position <= self.siteResidue.id.position: continue
            #print "Residue type for key value of allrotmers.siteRotamers_D: ", resNbr.type ################################################################################################## DEBUG
            for nbrRotamerIx in allRotamers_G.siteRotamers_D[resNbr].singleRotamers_D.keys():
                useRotamer(resNbr, [allRotamers_G.siteRotamers_D[resNbr].singleRotamers_D[nbrRotamerIx].rotamer])
                allRotamers_G.siteRotamers_D[resNbr].singleRotamers_D[nbrRotamerIx].putInHydrogens()
                energy = lj.residuePairEnergy(self.siteResidue, resNbr)
                rotamerPairEnergy_D_G[(self.siteResidue.id.position, self.rotamerIndex, resNbr.id.position, nbrRotamerIx)] = energy

                # Printout for debugging purposes:
                if energy != 0.0:
                    lineCount_G += 1
                    if lineCount_G/100*100 == lineCount_G:
                        print lineCount_G, "This site: ", self.siteResidue.type, self.siteResidue.id.position, "Rotamer ID: ", self.rotamerIndex, "Nbr site: ", resNbr.type, resNbr.id.position, "Rotamer ID: ", nbrRotamerIx, "Energy: ", energy    ###########################################################




                    
#==============================================================================
# SiteRotamers CLASS
#==============================================================================

class SiteRotamers(object):
    def __init__(self, res):
        global intrinsicEnergyThreshold_G
        self.res = res
        # Rotamers not needed later may be deleted so we track them with a dictionary:
           # Key is the original rotamer list index.
           # Value is a SingleRotamer object.
        self.singleRotamers_D = {}
        
        if res.type == "GLY" or res.type == "ALA":
            self.rotCount = 0
            return
        
        rot_L = getRotamers(res)[1]
        self.rotCount = len(rot_L)

        p0 = rot_L[0].rotamerProb

        # Generate all the singleRotamer objects and set them up in the singleRotamers_D.
        for i in range(len(rot_L)):
            probThisRot = rot_L[i].rotamerProb
            if probThisRot > 0.0000000000001:
                statSelf_Energy = 10.0 * numpy.log(p0/probThisRot)  # See Kingsford's thesis, pg. 33
                print res.type, "Rotamer index: ", i, "Statistical energy: ", statSelf_Energy ##########################################################
                singleRot = SingleRotamer(i, res, statSelf_Energy)
                if singleRot.intrinsicEnergy < intrinsicEnergyThreshold_G:
                    self.singleRotamers_D[i] = singleRot
                else:
                    print res.type, res.id.position, "Deleting rotamer ", singleRot.rotamerIndex, " with intrinsic energy: ", singleRot.intrinsicEnergy
                    del singleRot


    # -------------------------------------------------------------------------------------------
    # This function is needed by doGoldsteinDEE
    def __getMinDiffEnergyValues(self, rotAix, rotBix, resNbr):
        # Function to compute the min[] summand of the Goldstein criterion (see Equation 4 of the Goldstein
        # paper in the Biophysics Journal.
        # Input:
        #       rotAix:  index for rotamer A at this residue site
        #       rotBix:  index for rotamer B at this residue site
        #       resNbr:  residue neighbour (designated by index j in the Goldstein paper equation 4)

        # First collect the energy difference values in a list:
        # Don't forget: the rotamerPairEnergy_D_G dictionary has a tuple key with residue positions
        # in increasing order.
        global allRotamers_G
        
        eDiffVals_L = []
        for rotXix in allRotamers_G.siteRotamers_D[resNbr].singleRotamers_D.keys():
            if resNbr.id.position < self.res.id.position:
                eDiffVals_L.append(rotamerPairEnergy_D_G[(resNbr.id.position, rotXix, self.res.id.position, rotAix)] -
                                   rotamerPairEnergy_D_G[(resNbr.id.position, rotXix, self.res.id.position, rotBix)])
            else:
                eDiffVals_L.append(rotamerPairEnergy_D_G[(self.res.id.position, rotAix, resNbr.id.position, rotXix)] -
                                   rotamerPairEnergy_D_G[(self.res.id.position, rotBix, resNbr.id.position, rotXix)])
        return min(eDiffVals_L)

            
    # -------------------------------------------------------------------------------------------
    # Function to calculate the RHS of equation (4) in Goldstein' paper.    
    def goldsteinCriterionValue(self, rotAix, rotBix):        
        # Input:
        #       rotAix:  index for rotamer A at this residue site
        #       rotBix:  index for rotamer B at this residue site
        # If the return value is positive then we can delete the rotamer indexed by rotAix.
        sumMinDiffEnergies = 0.0
        for rNbr in nbrsMap_D_G[self.res]:
            if rNbr == self.res: continue
            if not rNbr in residuesWithRotamers_L_G: continue
            sumMinDiffEnergies += self.__getMinDiffEnergyValues(rotAix, rotBix, rNbr)
        return self.singleRotamers_D[rotAix].intrinsicEnergy - self.singleRotamers_D[rotBix].intrinsicEnergy + sumMinDiffEnergies

        
    # -------------------------------------------------------------------------------------------
    # Do dead end elimination for this residue position following the Goldstein strategy.
    # Instead of comparing every totamer at this site with every other rotamer, we pick the
    # rotamer with the lowest intrinsic energy and test to see if any other rotamer at this site
    # can be eliminated.
    def doGoldsteinDEEcomparingWithLowEnergyRotomer(self):
        # Find rotamer with lowest intrinsic energy:
        lowestEvalue = 99999999999999999.
        for rotIx in self.singleRotamers_D.keys():
            if self.singleRotamers_D[rotIx].intrinsicEnergy < lowestEvalue:
                lowestEvalue = self.singleRotamers_D[rotIx].intrinsicEnergy
                lowErotIx = rotIx
                
        for rotIx in self.singleRotamers_D.keys():
            if rotIx == lowErotIx: continue
            if self.goldsteinCriterionValue(rotIx, lowErotIx) > 0.0:
                print "Killing rotamer: ", rotIx, "at residue position: ", self.res.id.position
                self.singleRotamers_D.pop(rotIx)
                
    # -------------------------------------------------------------------------------------------
    # Do dead end elimination for this residue position following the Goldstein strategy.
    # After applying doGoldsteinDEEcomparingWithLowEnergyRotomer(self) we run this
    # algorithm which compares each rotamer with all sibling rotamers.
    def doGoldsteinDEE_AllRotomersVsAllRotamers(self):
        rotamersDeleted = False
        victims_S = Set()
        for rotAix in self.singleRotamers_D.keys():
            for rotBix in self.singleRotamers_D.keys():
                if rotAix == rotBix: continue
                if self.goldsteinCriterionValue(rotAix, rotBix) > 0.0:
                    victims_S.add(rotAix)
                
        for rotIx in victims_S:
            print "Killing rotamer: ", rotIx, "at residue position: ", self.res.id.position
            self.singleRotamers_D.pop(rotIx)

        if len(victims_S) > 0: rotamersDeleted = True
        return rotamersDeleted
                
#==============================================================================
# AllRotamers CLASS
#==============================================================================

class AllRotamers(object):
    def __init__(self, res_L):
        self.res_L = res_L
        
        # Dictionary of SiteRotamers object:
           # Key is a residue.
           # Value is a SiteRotamers object.
        self.siteRotamers_D = {}

        for r in res_L:
            self.siteRotamers_D[r] = SiteRotamers(r)
            
            # Now that siteRotamer initialization is done, leave the residue site
            # with the lowest energy rotamer:
            minEnergy = intrinsicEnergyThreshold_G
            for rotIx in self.siteRotamers_D[r].singleRotamers_D.keys():
                if self.siteRotamers_D[r].singleRotamers_D[rotIx].intrinsicEnergy < minEnergy:
                    minEnergy = self.siteRotamers_D[r].singleRotamers_D[rotIx].intrinsicEnergy
                    minEnergyRotIx = rotIx
            useRotamer(r, [self.siteRotamers_D[r].singleRotamers_D[minEnergyRotIx].rotamer])
            self.siteRotamers_D[r].singleRotamers_D[minEnergyRotIx].putInHydrogens()
            
            
    # -------------------------------------------------------------------------------------------
    # 
    def doEnergyCalculations(self):
        for r in self.res_L:
            for rotIx in self.siteRotamers_D[r].singleRotamers_D.keys():
                self.siteRotamers_D[r].singleRotamers_D[rotIx].updateRotPairEnergyDictionary()
                
                
    # -------------------------------------------------------------------------------------------
    # 
    def doGoldsteinLowEnergyRotamer(self):
        for sR in self.siteRotamers_D.keys():
            self.siteRotamers_D[sR].doGoldsteinDEEcomparingWithLowEnergyRotomer()
            
            
    # -------------------------------------------------------------------------------------------
    # 
    def doGoldsteinAllRotamersVsAllRotamers(self):
        rotamers_Deleted = True
        while rotamers_Deleted:
            for sR in self.siteRotamers_D.keys():
                rotamers_Deleted = self.siteRotamers_D[sR].doGoldsteinDEE_AllRotomersVsAllRotamers()
            
#============================================================================================
# MAINLINE
#============================================================================================


#targetPath = "1TIM"
#targetPath = "1c9o"
targetPath = "1CRN"
#targetPath = "1BRF"
#targetPath = "1AAC"
#targetPath = "1AHO"
#targetPath = "1B9O"
#targetPath = "1C5E"
#targetPath = "1CC7"
#targetPath = "1CEX"
#targetPath = "1CKU"
#targetPath = "1CTJ"
#targetPath = "1CZ9"
#targetPath = "1CZP"
#targetPath = "1D4T"
#targetPath = "1IGD"
#targetPath = "1MFM"
#targetPath = "1PLC"
#targetPath = "1QJ4"
#targetPath = "1QTN"
#targetPath = "1QU9"
#targetPath = "1RCF"
#targetPath = "1VFY"
#targetPath = "2PTH"
#targetPath = "3LZT"  # Skipped because several sidechains have "alternates". 
#targetPath = "4RXN"
#targetPath = "5P21"
#targetPath = "7RSA"  # Skipped because several sidechains have "alternates".

#targetPath = "6PAX"
#targetPath = "1BEC"
#targetPath = "1IFC"

#outputFile = file("Results/" + targetPath + "data.txt", 'w')
outputFile = file(targetPath + "data.txt", 'w')
outFormat = '%5d  %5d  %5d  %5d  %5d  %7.6f\n'

# The fixed residues either have no rotamers or have a conformation that is not to be replaced
# by a rotamer (eg. cysteine with a bridge).
typesOfFixedResidues_G = ('GLY', 'ALA')
intrinsicEnergyThreshold_G = 1000.0

# Dictionary of rotamer pair energies.  Key is a tuple: (residue position, rotamer number, residue position, rotamer number).
# Value is the corresponding pair energy.
rotamerPairEnergy_D_G = {}

runCommand("close session")

# Global variables:
#protein_G = chimera.openModels.open(targetPath + ".pdb", type="PDB")[0]
protein_G = chimera.openModels.open(targetPath, type="PDB")[0]

runCommand("sel @.B")
runCommand("del sel")

#threshold_G = 22.0
#threshold_G = 16.0
threshold_G = 10.0 #######################################################################################################################

lineCount_G = 0
#maxEnergyThreshold_G = 500.0


runCommand('set bg_color white')
runCommand('~ribbon')
runCommand('represent stick')
runCommand('display')

lj = LJ_evaluation()


# Build global dictionary that maps a residue (capable of having rotamers) to a list of neighbouring residues.
nbrsMap_D_G = {}
betaAtoms_L = []
for r in protein_G.residues:
    if validResidue(r):
        if r.findAtom('CB'): betaAtoms_L.append(r.findAtom('CB'))
        
shell_G = Shell(betaAtoms_L, 1.0, threshold_G) # We can now use shell_G.getAtomsInShell(centerAtom) to get atom neighbors

for r in protein_G.residues:
    if validResidue(r):
        nbrs_L = []
        cBForThisRes = r.findAtom('CB')
        if cBForThisRes:
            print "CB Res", cBForThisRes.name, cBForThisRes.residue.type
            for cb in shell_G.getAtomsInShell(cBForThisRes):
                nbrs_L.append(cb.residue)
            print "residue Position: ", r.id.position, "has nearby residues:"
            for rs in nbrs_L:
                print rs.type, rs.id.position    #################################### DEBUG
        nbrsMap_D_G[r] = nbrs_L


fixedRes_L_G = []
residuesWithRotamers_L_G = []
for r in protein_G.residues:
    if r.type in typesOfFixedResidues_G:
        fixedRes_L_G.append(r)
    else:
        if validResidue(r): residuesWithRotamers_L_G.append(r)

allRotamers_G = AllRotamers(residuesWithRotamers_L_G)
allRotamers_G.doEnergyCalculations()

# Record rotamer statistics prior to DEE:
#outputSurvivalFile = file("Results/" + targetPath + "survivorCount.txt", 'w')
outputSurvivalFile = file(targetPath + "survivorCount.txt", 'w')
outSurvivalFormat = '%5d %4s %5d\n'
outputSurvivalFile.write("Prior to DEE:\n")
totalRotCount = 0
for res in allRotamers_G.siteRotamers_D.keys():
    rotamer_Count = len(allRotamers_G.siteRotamers_D[res].singleRotamers_D.keys())
    totalRotCount += rotamer_Count
    line_Out = outSurvivalFormat % (res.id.position, res.type, rotamer_Count)
    outputSurvivalFile.write(line_Out)
line_Out = "Total rotamer count: " + str(totalRotCount) + "   Total number of edges (not counting self-loops): " + str(lineCount_G)
outputSurvivalFile.write(line_Out)


# Now do the DEE:
allRotamers_G.doGoldsteinLowEnergyRotamer()
print "End of first pass"
allRotamers_G.doGoldsteinAllRotamersVsAllRotamers()


# Report results:
lineCount = 0
for res in allRotamers_G.siteRotamers_D.keys():
    for rotNum in allRotamers_G.siteRotamers_D[res].singleRotamers_D.keys():
        lineCount += 1
        intrinsic_energy = allRotamers_G.siteRotamers_D[res].singleRotamers_D[rotNum].intrinsicEnergy
        lineOut = outFormat % (lineCount, res.id.position, rotNum, res.id.position, rotNum, intrinsic_energy)
        outputFile.write(lineOut)
        print lineCount, res.id.position, rotNum, res.id.position, rotNum, intrinsic_energy
        for nearbyRes in nbrsMap_D_G[res]:
            if not nearbyRes in residuesWithRotamers_L_G: continue
            if nearbyRes == res: continue
            if nearbyRes.id.position < res.id.position: continue
            for nearbyRotNum in allRotamers_G.siteRotamers_D[nearbyRes].singleRotamers_D.keys():
                e = rotamerPairEnergy_D_G[res.id.position, rotNum, nearbyRes.id.position, nearbyRotNum]
                if abs(e) < 0.000001: continue
                lineCount += 1
                #if lineCount % 10 == 0: print "                 Line Count = ", lineCount
                lineOut = outFormat % (lineCount, res.id.position, rotNum, nearbyRes.id.position, nearbyRotNum, e)
                outputFile.write(lineOut)
                print lineCount, res.id.position, rotNum, nearbyRes.id.position, nearbyRotNum, e
outputFile.close()


# Record rotamer statistics after DEE:
outputSurvivalFile.write("\n\nAfter DEE:\n")
totalRotCount = 0
for res in allRotamers_G.siteRotamers_D.keys():
    rotamer_Count = len(allRotamers_G.siteRotamers_D[res].singleRotamers_D.keys())
    totalRotCount += rotamer_Count
    line_Out = outSurvivalFormat % (res.id.position, res.type, rotamer_Count)
    outputSurvivalFile.write(line_Out)
line_Out = "Total rotamer count: " + str(totalRotCount)+ "    Total number of edges (including self-loops): " + str(lineCount)
outputSurvivalFile.write(line_Out)
outputSurvivalFile.close()



