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
    


class Core1(GeometryBase):
    def __init__(self):
        super().__init__()
        self.energy_window = {}
        self.unit = 'A'
        self.geom_init()
        self.names = list(self.geom.keys())
        self.spin_init()
        self.charge_init()
        self.energy_window_init()
        pass


    def geom_init(self):
        self.geom['001_Be'] = '''
        Be 0.0000 0.0000 0.0000
        '''
        
        self.geom['002_C2H4'] = '''
        C1	0.0000	0.0000	0.6695
        C2	0.0000	0.0000	-0.6695
        H3	0.0000	0.9289	1.2321
        H4	0.0000	-0.9289	1.2321
        H5	0.0000	0.9289	-1.2321
        H6	0.0000	-0.9289	-1.2321
        '''
        
        self.geom['003_H2CO'] = '''
        O1	0.0000	0.0000	1.2050
        C2	0.0000	0.0000	0.0000
        H3	0.0000	0.9429	-0.5876
        H4	0.0000	-0.9429	-0.5876
        '''

        self.geom['004_C2H2'] = '''
        C1	0.0000	0.0000	0.6013
        C2	0.0000	0.0000	-0.6013
        H3	0.0000	0.0000	1.6644
        H4	0.0000	0.0000	-1.6644
        '''

        self.geom['005_HCN'] = '''
        C1	0.0000	0.0000	0.0000
        H2	0.0000	0.0000	1.0640
        N3	0.0000	0.0000	-1.1560
        '''

        self.geom['006_CO'] = '''
        C1	0.0000	0.0000	0.0000
        O2	0.0000	0.0000	1.1282
        '''

        self.geom['007_CH3OH'] = '''
        C1	-0.0503	0.6685	0.0000
        O2	-0.0503	-0.7585	0.0000
        H3	-1.0807	1.0417	0.0000
        H4	0.4650	1.0417	0.8924
        H5	0.4650	1.0417	-0.8924
        H6	0.8544	-1.0677	0.0000
        '''
        
        self.geom['008_CH4'] = '''
        C 0.0000 0.0000 0.0000
        H 0.6276 0.6276 0.6276
        H -0.6276 -0.6276 0.6276
        H -0.6276 0.6276 -0.6276
        H 0.6276 -0.6276 -0.6276
        '''

        self.geom['009_NH3'] = '''
        N1	0.0000	0.0000	0.0000
        H2	0.0000	-0.9377	-0.3816
        H3	0.8121	0.4689	-0.3816
        H4	-0.8121	0.4689	-0.3816
        '''

        self.geom['010_N2'] = '''
        N1	0.0000	0.0000	0.5488
        N2	0.0000	0.0000	-0.5488
        '''

        self.geom['011_H2O'] = '''
        O1	0.0000	0.0000	0.1173
        H2	0.0000	0.7572	-0.4692
        H3	0.0000	-0.7572	-0.4692
        '''

        self.geom['012_F2'] = '''
        F1	0.0000	0.0000	0.0000
        F2	0.0000	0.0000	1.4119
        '''

        self.geom['013_FH'] = '''
        F1	0.0000	0.0000	0.0000
        H2	0.0000	0.0000	0.9168
        '''

        self.geom['014_Ne'] = '''
        Ne	0.0000	0.0000	0.0000
        '''

        self.geom['015_N2O'] = '''
        N1	0.0000	0.0000	-1.1998
        N2	0.0000	0.0000	-0.0716
        O3	0.0000	0.0000	1.1126
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
    

    def energy_window_init(self):
        for name in ['001_Be']:
            self.energy_window[name] = [95, 130]
        for name in ['002_C2H4', '004_C2H2', '008_CH4']:
            self.energy_window[name] = [250, 310]
        for name in ['009_NH3', '010_N2', '015_N2O']:
            self.energy_window[name] = [370, 425]
        for name in ['005_HCN']:
            self.energy_window[name] = [250, 425]
        for name in ['003_H2CO','006_CO', '007_CH3OH']:
            self.energy_window[name] = [250, 560]
        for name in ['011_H2O']:
            self.energy_window[name] = [500, 560]
        for name in ['012_F2', '013_FH']:
            self.energy_window[name] = [640, 710]
        for name in ['014_Ne']:
            self.energy_window[name] = [820, 890]
        return None
    

def check():
    geom = Core1()
    names = geom.get_names()
    print('unit', geom.unit)
    print('number of molecules:', len(names))
    for name in names:
        print('name:', name)
        mol = geom.get_molecule(name, basis='aug-cc-pvtz')
        print(mol.basis)
        print(mol.natm)
        print(mol.nao)
        print(geom.energy_window[name])
    return None


if __name__ == '__main__':
    check()
    pass