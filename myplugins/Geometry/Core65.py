import pyscf
from pyscf import gto


class GeometryBase():
    def __init__(self):
        self.names = []
        self.unit = 'A'
        self.geom = {}
        self.spin = {}
        self.charge = {}
        pass

    def get_names(self):
        return self.names

    def get_geom(self, name):
        return self.geom[name]
    
    def get_spin(self, name):
        return self.spin[name]
    
    def get_charge(self, name):
        return self.charge[name]

    def get_molecule(self, name, 
                     basis='ccpvtz', 
                     charge=-999999,
                     spin=-999999,
                     symmetry=False, 
                     verbose=3):
        if name not in self.names:
            raise ValueError(f"Name {name} not found in geometry database.")
        if charge == -999999:
            charge = self.get_charge(name)
        if spin == -999999:
            spin = self.get_spin(name)
        mol = gto.M(
                unit = self.unit,
                atom = self.get_geom(name),
                charge = charge,
                spin = spin,
                basis = basis,
                symmetry = symmetry,
                verbose = verbose
                )
        return mol



class Core65_sub(GeometryBase):

    def __init__(self):
        super().__init__()
        self.unit = 'A'
        self.geom_init()
        self.names = list(self.geom.keys())
        self.spin_init()
        self.charge_init()
        pass


    def geom_init(self):
        self.geom['CH4'] = '''
        C  0.00000016  0.00001918  0.00000016  
        H  0.63232625  -0.63226771  0.63232625  
        H  -0.63232571  0.63233315  0.63230028  
        H  -0.63230098  -0.63231778  -0.63230098  
        H  0.63230028  0.63233315  -0.63232571 
        '''

        self.geom['C2H2'] = '''
        C  -0.00000000  0.00000000  0.60316922  
        C  0.00000000  0.00000000  -0.60316922  
        H  0.00000000  -0.00000000  1.67312311  
        H  0.00000000  -0.00000000  -1.67312311
        '''

        self.geom['CO'] = '''
        C  -0.00000000  0.00000000  0.07378202  
        O  0.00000000  0.00000000  1.20921798
        '''

        self.geom['H2O'] = '''
        O  0.00000000  -0.00000000  -0.00614048  
        H  0.76443318  -0.00000000  0.58917024  
        H  -0.76443318  0.00000000  0.58917024 
        '''

        self.geom['NH3'] = '''
        N  -0.00000000  0.00003004  0.00668480  
        H  0.00000000  -0.94342500  -0.38383647  
        H  0.81706006  0.47174748  -0.38382417  
        H  -0.81706006  0.47174748  -0.38382417 
        '''

        self.geom['N2'] = '''
        N  0.00000000  0.00000000  -0.00240360  
        N  -0.00000000  -0.00000000  1.10010360 
        '''
        return None
    
    def spin_init(self):
        for name in self.names:
            self.spin[name] = 0
        return None
    
    def charge_init(self):
        for name in self.names:
            self.charge[name] = 0
        return None
    

def check():
    geom = Core65_sub()
    names = geom.get_names()
    print('unit', geom.unit)
    print('number of molecules:', len(names))
    for name in names:
        print('name:', name)
        mol = geom.get_molecule(name, basis='def2-tzvp')
        print(mol.natm)
        print(mol.nao)
    return None


if __name__ == '__main__':
    check()
    pass