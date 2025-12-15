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



class NHTBH38(GeometryBase):

    def __init__(self):
        super().__init__()
        self.unit = 'A'
        self.geom_init()
        self.names = list(self.geom.keys())
        # self.spin_init()
        # self.charge_init()
        pass


    def geom_init(self):
        self.geom['C2H4'] = '''
        C       -0.000000    0.000000    0.665593
        C        0.000000   -0.000000   -0.665593
        H        0.000000    0.921495    1.231668
        H       -0.000000   -0.921495    1.231668
        H        0.000000    0.921495   -1.231668
        H       -0.000000   -0.921495   -1.231668
        '''
        self.charge['C2H4'] = 0
        self.spin['C2H4'] = 0

        self.geom['CH3CH2CH2'] = '''
        C        1.208440   -0.287189    0.000057
        C       -0.065359    0.576132   -0.000057
        C       -1.314787   -0.239518   -0.000011
        H        1.241369   -0.928395    0.881234
        H        1.241394   -0.928586   -0.880980
        H        2.101871    0.338727    0.000000
        H       -0.048218    1.226851   -0.877089
        H       -0.048272    1.227037    0.876834
        H       -1.729146   -0.615771    0.924435
        H       -1.728763   -0.616415   -0.924369
        '''
        self.charge['CH3CH2CH2'] = 0
        self.spin['CH3CH2CH2'] = 1

        self.geom['CH3CH2'] = '''
        C       -0.258719   -0.816829    0.000000
        C       -0.250987    0.674191    0.000000
        H        0.758830   -1.225939    0.000000
        H       -0.758830   -1.213866    0.883419
        H       -0.758830   -1.213866   -0.883419
        H       -0.170021    1.225939   -0.924320
        H       -0.170021    1.225939    0.924320
        '''
        self.charge['CH3CH2'] = 0
        self.spin['CH3CH2'] = 1

        self.geom['CH3Cl'] = '''
        C        0.000000    0.000000   -1.125886
        Cl       0.000000    0.000000    0.656830
        H        0.000000    1.027993   -1.470264
        H        0.890268   -0.513997   -1.470264
        H       -0.890268   -0.513997   -1.470264
        '''
        self.charge['CH3Cl'] = 0
        self.spin['CH3Cl'] = 0

        self.geom['CH3'] = '''
        C                  0.00000000    0.00000000    0.00000000
        H                  0.00000000    0.00000000    1.07731727
        H                  0.93298412    0.00000000   -0.53865863
        H                 -0.93298412   -0.00000000   -0.53865863
        '''
        self.charge['CH3'] = 0
        self.spin['CH3'] = 1

        self.geom['CH3F'] = '''
        C       -0.632074    0.000001   -0.000000
        F        0.749117    0.000002   -0.000002
        H       -0.983182   -0.338489    0.972625
        H       -0.983222    1.011553   -0.193172
        H       -0.983203   -0.673084   -0.779437
        '''
        self.charge['CH3F'] = 0
        self.spin['CH3F'] = 0

        self.geom['CH3OH'] = '''
        C       -0.046423    0.663069    0.000000
        O       -0.046423   -0.755063    0.000000
        H       -1.086956    0.975938    0.000000
        H        0.860592   -1.057039    0.000000
        H        0.438145    1.071594    0.889539
        H        0.438145    1.071594   -0.889539
        '''
        self.charge['CH3OH'] = 0
        self.spin['CH3OH'] = 0

        self.geom['ClCH3Clcomp'] = '''
        Cl        .000000     .000000   -2.384735
        C         .000000     .000000    -.566331
        H         .000000    1.025066    -.224379
        H        -.887734    -.512533    -.224379
        H         .887734    -.512533    -.224379
        Cl        .000000     .000000    2.624213
        '''
        self.charge['ClCH3Clcomp'] = -1
        self.spin['ClCH3Clcomp'] = 0

        self.geom['Cl-'] = '''
        Cl        0.000000    0.000000    0.000000
        '''
        self.charge['Cl-'] = -1
        self.spin['Cl-'] = 0

        self.geom['Cl'] = '''
        Cl        0.000000    0.000000    0.000000
        '''
        self.charge['Cl'] = 0
        self.spin['Cl'] = 1

        self.geom['CO'] = '''
        O
        C 1 1.12960815
        '''
        self.charge['CO'] = 0
        self.spin['CO'] = 0

        self.geom['F2'] = '''
        F
        F,1,1.3952041
        '''
        self.charge['F2'] = 0
        self.spin['F2'] = 0

        self.geom['FCH3Clcomp1'] = '''
        Cl        .000000     .000000    1.623138
        C         .000000     .000000    -.227358
        H         .000000    1.026321    -.555141
        H         .888820    -.513160    -.555141
        H        -.888820    -.513160    -.555141
        F         .000000     .000000   -2.729308
        '''
        self.charge['FCH3Clcomp1'] = -1
        self.spin['FCH3Clcomp1'] = 0

        self.geom['FCH3Clcomp2'] = '''
        F         .000000     .000000   -2.648539
        C         .000000     .000000   -1.240170
        H         .000000    1.024719    -.886406
        H        -.887432    -.512359    -.886406
        H         .887432    -.512359    -.886406
        Cl        .000000     .000000    1.996299
        '''
        self.charge['FCH3Clcomp2'] = -1
        self.spin['FCH3Clcomp2'] = 0

        self.geom['FCH3Fcomp'] = '''       
        F         .000000     .000000   -1.847626
        C         .000000     .000000    -.421873
        H         .000000    1.023581    -.073843
        H        -.886447    -.511791    -.073843
        H         .886447    -.511791    -.073843
        F         .000000     .000000    2.153489
        '''
        self.charge['FCH3Fcomp'] = -1
        self.spin['FCH3Fcomp'] = 0

        self.geom['FCl'] = '''
        F
        Cl,1,1.63033021
        '''
        self.charge['FCl'] = 0
        self.spin['FCl'] = 0

        self.geom['F-'] = '''
        F         .000000     .000000    0.000000
        '''
        self.charge['F-'] = -1
        self.spin['F-'] = 0

        self.geom['F'] = '''
        F         .000000     .000000    0.000000
        '''
        self.charge['F'] = 0
        self.spin['F'] = 1

        self.geom['HCl'] = '''
        Cl
        H,1,1.27444789
        '''
        self.charge['HCl'] = 0
        self.spin['HCl'] = 0

        self.geom['HCN'] = '''
        C        0.000000    0.000000   -0.500365
        N        0.000000    0.000000    0.652640
        H        0.000000    0.000000   -1.566291
        '''
        self.charge['HCN'] = 0
        self.spin['HCN'] = 0

        self.geom['HCO'] = '''
        H        -.009057     .000000    -.007086
        C        -.007035     .000000    1.109678
        O         .956040     .000000    1.785656
        '''
        self.charge['HCO'] = 0
        self.spin['HCO'] = 1


        self.geom['H'] = '''
        H         .000000     .000000    0.000000
        '''
        self.charge['H'] = 0
        self.spin['H'] = 1

        self.geom['HF'] = '''
        F
        H,1,0.91538107
        '''
        self.charge['HF'] = 0
        self.spin['HF'] = 0

        self.geom['HN2'] = '''
        N                  0.00000000    0.00000000    0.00000000
        N                  0.00000000    0.00000000    1.17820000
        H                  0.93663681    0.00000000    1.64496947
        '''        
        self.charge['HN2'] = 0
        self.spin['HN2'] = 1

        self.geom['HNC'] = '''
        C         .000000     .000000    -.737248
        N         .000000     .000000     .432089
        H         .000000     .000000    1.426960
        '''
        self.charge['HNC'] = 0
        self.spin['HNC'] = 0

        self.geom['HOCH3Fcomp1'] = '''
        C       -1.297997    -.389518    -.000034
        O       -.477223      .728021     .000054
        H       -2.351922    -.080232    -.008639
        H       -1.140853   -1.035821    -.878101
        H       -1.153178   -1.027513     .886359
        H         .510580     .371160     .000243
        F        1.749016    -.190517    -.000010
        '''
        self.charge['HOCH3Fcomp1'] = -1
        self.spin['HOCH3Fcomp1'] = 0

        self.geom['HOCH3Fcomp2'] = '''
        F         .000371   -2.468340     .021390
        C        -.276642   -1.074418    -.002690
        H         .649290    -.516500    -.009016
        H        -.841989    -.847119    -.897075
        H        -.851028    -.826589     .881417
        O        -.301713    1.582524    -.206544
        H        -.605112    2.492434    -.164305
        '''
        self.charge['HOCH3Fcomp2'] = -1
        self.spin['HOCH3Fcomp2'] = 0

        self.geom['N2'] = '''
        N
        N,1,1.09710935
        '''
        self.charge['N2'] = 0
        self.spin['N2'] = 0

        self.geom['N2O'] = '''
        N                  0.00000000    0.00000000    0.00000000
        N                  0.00000000    0.00000000    1.12056262
        O                  0.00000000    0.00000000    2.30761092
        '''
        self.charge['N2O'] = 0
        self.spin['N2O'] = 0

        self.geom['OH-'] = '''
        O
        H,1,0.96204317
        '''
        self.charge['OH-'] = -1
        self.spin['OH-'] = 0

        self.geom['OH'] = '''
        O
        H,1,0.96889819
        '''
        self.charge['OH'] = 0
        self.spin['OH'] = 1

        self.geom['TS11'] = '''
        F         .000000     .000000    -2.537929
        C         .000000     .000000    -.488372
        H         .000000    1.062087    -.614972
        H        -.919795    -.531044    -.614972
        H         .919795    -.531044    -.614972
        Cl        .000000     .000000    1.624501
        '''
        self.charge['TS11'] = -1
        self.spin['TS11'] = 0

        self.geom['TS13'] = '''
        F         .022536    -.007453     .005529
        C        -.018420     .005037    1.764925
        H        1.048050     .005240    1.854146
        H        -.547819     .934707    1.792224
        H        -.548955    -.923433    1.805762
        O         .001265     .019200    3.750599
        H        -.926763     .031615    3.997581
        '''
        self.charge['TS13'] = -1
        self.spin['TS13'] = 0

        self.geom['TS15'] = '''
        N                  0.00000000    0.00000000    0.00000000
        N                  0.00000000    0.00000000    1.12281100
        H                  1.26844651    0.00000000    1.78433286
        '''
        self.charge['TS15'] = 0
        self.spin['TS15'] = 1

        self.geom['TS16'] = '''
        H       -1.520864    1.388829     .000000
        C         .108633     .549329     .000000
        O         .108633    -.585601     .000000
        '''
        self.charge['TS16'] = 0
        self.spin['TS16'] = 1

        self.geom['TS17'] = '''
        C       -0.567877    0.000051   -0.218958 
        C        0.751139   -0.000036    0.041932 
        H       -1.493884   -0.000488    1.531765 
        H       -1.101691    0.920651   -0.408626 
        H       -1.102022   -0.920234   -0.409110 
        H        1.299128   -0.922344    0.173763 
        H        1.298899    0.922325    0.174363
        '''
        self.charge['TS17'] = 0
        self.spin['TS17'] = 1

        self.geom['TS18'] = '''
        C       -0.472132    0.645933   -0.000043
        C       -1.382617   -0.363885   -0.000002
        H       -0.232044    1.164575   -0.917264
        H       -0.232342    1.164759    0.917169
        H       -1.727128   -0.809810    0.922519
        H       -1.726936   -0.810131   -0.922435
        C        1.612015   -0.242189    0.000035
        H        2.195182    0.668671   -0.001269
        H        1.589423   -0.809619   -0.918632
        H        1.590245   -0.807598    0.919969
        '''
        self.charge['TS18'] = 0
        self.spin['TS18'] = 1

        self.geom['TS19'] = '''
        C         .080319     .620258     .000000
        N         .080319    -.568095     .000000
        H       -1.044148     .255121     .000000
        '''
        self.charge['TS19'] = 0
        self.spin['TS19'] = 0

        self.geom['TS1'] = '''
        H        -.303286   -1.930712     .000000
        O        -.861006    -.621526     .000000
        N         .000000     .257027     .000000
        N        1.027333     .729104     .000000
        '''
        self.charge['TS1'] = 0
        self.spin['TS1'] = 1

        self.geom['TS2'] = '''
        H        0.000000    0.000000    1.137217
        F        0.000000    0.000000    0.000000
        H        0.000000    0.000000   -1.137217
        '''
        self.charge['TS2'] = 0
        self.spin['TS2'] = 1

        self.geom['TS3'] = '''
        H        0.000000    0.000000    1.485800
        Cl       0.000000    0.000000    0.000000
        H        0.000000    0.000000   -1.485800
        '''
        self.charge['TS3'] = 0
        self.spin['TS3'] = 1

        self.geom['TS4'] = '''
        H        -.039764     .000000     .044106
        F        -.049321     .000000    1.282554
        C        -.061544     .000000    2.951157
        H         .990497     .000000    3.194275
        H        -.590070     .912355    3.183481
        H        -.590070    -.912355    3.183481
        '''
        self.charge['TS4'] = 0
        self.spin['TS4'] = 1

        self.geom['TS5'] = '''
        H        0.000000    0.000000   -2.231273
        F        0.000000    0.000000   -0.616218
        F        0.000000    0.000000    0.864138
        '''
        self.charge['TS5'] = 0
        self.spin['TS5'] = 1

        self.geom['TS6'] = '''
        Cl       1.454749   -0.001237   -0.000040
        F       -0.323587    0.004631    0.000124
        C       -2.387418   -0.002147   -0.000073
        H       -2.495086   -0.855361   -0.649404
        H       -2.497313   -0.138673    1.063139
        H       -2.501537    0.986269   -0.413734
        '''
        self.charge['TS6'] = 0
        self.spin['TS6'] = 1

        self.geom['TS7'] = '''
        F         .003098    -.018892    -.015456
        C        -.000149    -.000140    1.807857
        H        1.069449     .001708    1.809761
        H        -.536607     .925133    1.796935
        H        -.532601    -.927783    1.817058
        F        -.003191     .019974    3.631845
        '''
        self.charge['TS7'] = -1
        self.spin['TS7'] = 0

        self.geom['TS9'] = '''
        Cl       2.322581    -.000132     .000140
        C        -.000085     .000491    -.000509
        H         .000077    -.744290    -.767605
        H        -.000320    -.291443    1.028021
        H         .000081    1.037218    -.261959
        Cl      -2.322542    -.000129     .000130
        '''
        self.charge['TS9'] = -1
        self.spin['TS9'] = 0
        return None
    
    def calculate_bh(self, energies:dict, unit='kcal'):
        bhs = {}
        if unit == 'kcal':
            e_factor = 627.5095
        elif unit == 'kJ':
            e_factor = 2625.4996
        elif unit == 'eV':
            e_factor = 27.2114
        elif unit == 'au':
            e_factor = 1.0
        else:
            raise ValueError(f"Unit {unit} not recognized. Use 'au', 'kcal', 'kJ', or 'eV'.")
        # Reaction 1: H + N2O → N2 + OH
        bhs['F1'] = (energies['TS1'] - energies['H'] - energies['N2O']) * e_factor
        bhs['R1'] = (energies['TS1'] - energies['N2'] - energies['OH']) * e_factor
        # Reaction 2: H + HF → HF + H
        bhs['F2'] = (energies['TS2'] - energies['H'] - energies['HF']) * e_factor
        bhs['R2'] = (energies['TS2'] - energies['HF'] - energies['H']) * e_factor
        # Reaction 3: H + HCl → HCl + H
        bhs['F3'] = (energies['TS3'] - energies['H'] - energies['HCl']) * e_factor
        bhs['R3'] = (energies['TS3'] - energies['HCl'] - energies['H']) * e_factor
        # Reaction 4: H + CH3F → HF + CH3
        bhs['F4'] = (energies['TS4'] - energies['H'] - energies['CH3F']) * e_factor
        bhs['R4'] = (energies['TS4'] - energies['HF'] - energies['CH3']) * e_factor
        # Reaction 5: H + F2 → HF + F
        bhs['F5'] = (energies['TS5'] - energies['H'] - energies['F2']) * e_factor
        bhs['R5'] = (energies['TS5'] - energies['HF'] - energies['F']) * e_factor
        # Reaction 6: CH3 + FCl → CH3F + Cl
        bhs['F6'] = (energies['TS6'] - energies['CH3'] - energies['FCl']) * e_factor
        bhs['R6'] = (energies['TS6'] - energies['CH3F'] - energies['Cl']) * e_factor
        # Reaction 7: F- + CH3F → CH3F + F-
        bhs['F7'] = (energies['TS7'] - energies['F-'] - energies['CH3F']) * e_factor
        bhs['R7'] = (energies['TS7'] - energies['CH3F'] - energies['F-']) * e_factor
        # Reaction 8: FCH3Fcomp → FCH3Fcomp
        bhs['F8'] = (energies['TS7'] - energies['FCH3Fcomp']) * e_factor
        bhs['R8'] = (energies['TS7'] - energies['FCH3Fcomp']) * e_factor
        # Reaction 9: Cl- + CH3Cl → CH3Cl + Cl-
        bhs['F9'] = (energies['TS9'] - energies['Cl-'] - energies['CH3Cl']) * e_factor
        bhs['R9'] = (energies['TS9'] - energies['CH3Cl'] - energies['Cl-']) * e_factor
        # Reaction 10: ClCH3Clcomp → ClCH3Clcomp
        bhs['F10'] = (energies['TS9'] - energies['ClCH3Clcomp']) * e_factor
        bhs['R10'] = (energies['TS9'] - energies['ClCH3Clcomp']) * e_factor
        # Reaction 11: F- + CH3Cl → CH3F + Cl-
        bhs['F11'] = (energies['TS11'] - energies['F-'] - energies['CH3Cl']) * e_factor
        bhs['R11'] = (energies['TS11'] - energies['CH3F'] - energies['Cl-']) * e_factor
        # Reaction 12: FCH3Clcomp1 → FCH3Clcomp2
        bhs['F12'] = (energies['TS11'] - energies['FCH3Clcomp1']) * e_factor
        bhs['R12'] = (energies['TS11'] - energies['FCH3Clcomp2']) * e_factor
        # Reaction 13: OH- + CH3F → CH3OH + F-
        bhs['F13'] = (energies['TS13'] - energies['OH-'] - energies['CH3F']) * e_factor
        bhs['R13'] = (energies['TS13'] - energies['CH3OH'] - energies['F-']) * e_factor
        # Reaction 14: HOCH3Fcomp2 → HOCH3Fcomp1
        bhs['F14'] = (energies['TS13'] - energies['HOCH3Fcomp2']) * e_factor
        bhs['R14'] = (energies['TS13'] - energies['HOCH3Fcomp1']) * e_factor
        # Reaction 15: N2 + H → HN2
        bhs['F15'] = (energies['TS15'] - energies['N2'] - energies['H']) * e_factor
        bhs['R15'] = (energies['TS15'] - energies['HN2']) * e_factor
        # Reaction 16: H + CO → HCO
        bhs['F16'] = (energies['TS16'] - energies['H'] - energies['CO']) * e_factor
        bhs['R16'] = (energies['TS16'] - energies['HCO']) * e_factor
        # Reaction 17: C2H4 + H → CH3CH2
        bhs['F17'] = (energies['TS17'] - energies['C2H4'] - energies['H']) * e_factor
        bhs['R17'] = (energies['TS17'] - energies['CH3CH2']) * e_factor
        # Reaction 18: C2H4 + CH3 → CH3CH2CH2
        bhs['F18'] = (energies['TS18'] - energies['C2H4'] - energies['CH3']) * e_factor
        bhs['R18'] = (energies['TS18'] - energies['CH3CH2CH2']) * e_factor
        # Reaction 19: HCN → HNC
        bhs['F19'] = (energies['TS19'] - energies['HCN']) * e_factor
        bhs['R19'] = (energies['TS19'] - energies['HNC']) * e_factor
        return bhs
    
    def get_reference(self, unit='kcal'):
        bh_ref = {}
        e_factor = 1.0
        if unit == 'kcal':
            e_factor = 1.0
        elif unit == 'kJ':
            e_factor = 4.184
        elif unit == 'eV':
            e_factor = 0.0433641
        elif unit == 'au':
            e_factor = 0.00159362
        else:
            raise ValueError(f"Unit {unit} not recognized. Use 'au', 'kcal', 'kJ', or 'eV'.")
        # Reference barrier heights from the original paper (kcal/mol)
        bh_ref['F1'] = 18.14
        bh_ref['R1'] = 83.22
        bh_ref['F2'] = 42.18
        bh_ref['R2'] = 42.18
        bh_ref['F3'] = 18.00
        bh_ref['R3'] = 18.00
        bh_ref['F4'] = 30.38
        bh_ref['R4'] = 57.02
        bh_ref['F5'] = 2.27
        bh_ref['R5'] = 106.18
        bh_ref['F6'] = 7.43
        bh_ref['R6'] = 60.17
        bh_ref['F7'] = -0.34
        bh_ref['R7'] = -0.34
        bh_ref['F8'] = 13.38
        bh_ref['R8'] = 13.38
        bh_ref['F9'] = 3.10
        bh_ref['R9'] = 3.10
        bh_ref['F10'] = 13.61
        bh_ref['R10'] = 13.61
        bh_ref['F11'] = -12.54
        bh_ref['R11'] = 20.11
        bh_ref['F12'] = 2.89
        bh_ref['R12'] = 29.62
        bh_ref['F13'] = -2.78
        bh_ref['R13'] = 17.33
        bh_ref['F14'] = 10.96
        bh_ref['R14'] = 47.20
        bh_ref['F15'] = 14.69
        bh_ref['R15'] = 10.72
        bh_ref['F16'] = 3.17
        bh_ref['R16'] = 22.68
        bh_ref['F17'] = 1.72
        bh_ref['R17'] = 41.75
        bh_ref['F18'] = 6.85
        bh_ref['R18'] = 32.97
        bh_ref['F19'] = 48.16
        bh_ref['R19'] = 33.11
        for key in bh_ref:
            bh_ref[key] = bh_ref[key] * e_factor
        return bh_ref


class HTBH38(GeometryBase):

    def __init__(self):
        super().__init__()
        self.unit = 'A'
        self.geom_init()
        self.names = list(self.geom.keys())
        # self.spin_init()
        # self.charge_init()
        pass

    def geom_init(self):
        self.geom['C2H5'] = '''
        C                  0.00000000    0.00000000    0.00000000
        C                  0.00000000    0.00000000    1.49013941
        H                  1.01376730    0.00000000    1.89114067
        H                 -0.84854872    0.37413531   -0.55286471
        H                 -0.50105756   -0.88768462    1.89585167
        H                 -0.52500771    0.86748481    1.89104640
        H                  0.77218291   -0.51269721   -0.55357184
        '''
        self.charge['C2H5'] = 0
        self.spin['C2H5'] = 1

        self.geom['C2H6'] = '''
        C                  0.00000000    0.00000000    0.00000000
        C                  0.00000000    0.00000000    1.52618350
        H                  1.01606678    0.00000000    1.92140419
        H                  0.50959528   -0.87903863   -0.39521755
        H                 -0.50802900   -0.87994528    1.92140200
        H                 -0.50804062    0.87993794    1.92140070
        H                 -1.01606618   -0.00180333   -0.39521787
        H                  0.50646983    0.88084344   -0.39521724
        '''
        self.charge['C2H6'] = 0
        self.spin['C2H6'] = 0

        self.geom['C5H8'] = '''
        C       -2.055638   -0.612272    0.000007
        C       -1.231096    0.640448    0.000049
        C        0.105634    0.734273    0.000026
        C        1.057555   -0.374407   -0.000044
        C        2.383583   -0.198936   -0.000036
        H       -2.705085   -0.641597    0.877132
        H       -2.705129   -0.641508   -0.877089
        H       -1.451332   -1.516079   -0.000055
        H       -1.793665    1.567586    0.000103
        H        0.545756    1.725643    0.000064
        H        0.665262   -1.383242   -0.000105
        H        3.064689   -1.037719   -0.000088
        H        2.819275    0.792285    0.000023
        '''
        self.charge['C5H8'] = 0
        self.spin['C5H8'] = 0

        self.geom['CH3'] = '''
        C                  0.00000000    0.00000000    0.00000000
        H                  0.00000000    0.00000000    1.07731727
        H                  0.93298412    0.00000000   -0.53865863
        H                 -0.93298412   -0.00000000   -0.53865863
        '''
        self.charge['CH3'] = 0
        self.spin['CH3'] = 1

        self.geom['CH4'] = '''
        C                  0.00000000    0.00000000    0.00000000
        H                  0.00000000    0.00000000    1.08744517
        H                  1.02525314    0.00000000   -0.36248173
        H                 -0.51262658    0.88789525   -0.36248173
        H                 -0.51262658   -0.88789525   -0.36248173
        '''
        self.charge['CH4'] = 0
        self.spin['CH4'] = 0

        self.geom['Cl'] = '''
        Cl        0.000000    0.000000    0.000000
        '''
        self.charge['Cl'] = 0
        self.spin['Cl'] = 1

        self.geom['F'] = '''
        F         .000000     .000000    0.000000
        '''
        self.charge['F'] = 0
        self.spin['F'] = 1

        self.geom['H2'] = '''
        H
        H,1,0.74187646
        '''
        self.charge['H2'] = 0
        self.spin['H2'] = 0

        self.geom['H2O'] = '''
        O
        H,1,0.95691441
        H,1,0.95691441,2,104.51706026
        '''
        self.charge['H2O'] = 0
        self.spin['H2O'] = 0

        self.geom['H2S'] = '''
        S        0.000000    0.000000    0.102519
        H        0.000000    0.966249   -0.820154
        H        0.000000   -0.966249   -0.820154
        '''
        self.charge['H2S'] = 0
        self.spin['H2S'] = 0

        self.geom['HCl'] = '''
        Cl
        H,1,1.27444789
        '''
        self.charge['HCl'] = 0
        self.spin['HCl'] = 0

        self.geom['H'] = '''
        H         .000000     .000000    0.000000
        '''
        self.charge['H'] = 0
        self.spin['H'] = 1

        self.geom['HF'] = '''
        F
        H,1,0.91538107
        '''
        self.charge['HF'] = 0
        self.spin['HF'] = 0

        self.geom['HS'] = '''
        S
        H,1,1.34020229
        '''
        self.charge['HS'] = 0
        self.spin['HS'] = 1

        self.geom['NH2'] = '''
        N
        H,1,1.02404748
        H,1,1.02404748,2,103.15937043
        '''
        self.charge['NH2'] = 0
        self.spin['NH2'] = 1

        self.geom['NH3'] = '''
        N        0.000000    0.000000    0.112890
        H        0.000000    0.938024   -0.263409
        H        0.812353   -0.469012   -0.263409
        H       -0.812353   -0.469012   -0.263409
        '''
        self.charge['NH3'] = 0
        self.spin['NH3'] = 0

        self.geom['NH'] = '''
        N
        H,1,1.03673136
        '''
        self.charge['NH'] = 0
        self.spin['NH'] = 2

        self.geom['O'] = '''
        O         .000000     .000000    0.000000
        '''
        self.charge['O'] = 0
        self.spin['O'] = 2

        self.geom['OH'] = '''
        O
        H,1,0.96889819
        '''
        self.charge['OH'] = 0
        self.spin['OH'] = 1

        self.geom['PH2'] = '''
        P        0.000000    0.000000   -0.115657
        H        1.020130    0.000000    0.867427
        H       -1.020130    0.000000    0.867427
        '''
        self.charge['PH2'] = 0
        self.spin['PH2'] = 1

        self.geom['PH3'] = '''
        P        0.000000    0.000000    0.126411
        H        1.191339    0.000000   -0.632056
        H       -0.595669   -1.031730   -0.632056
        H       -0.595669    1.031730   -0.632056
        '''
        self.charge['PH3'] = 0
        self.spin['PH3'] = 0

        self.geom['TS10'] = '''
        C        0.000290   -1.142289    0.000000
        H       -1.055957   -1.384735    0.000000
        H        0.520167   -1.407389    0.912447
        H        0.520167   -1.407389   -0.912447
        H        0.011560    0.160099    0.000000
        O        0.000290    1.361643    0.000000
        '''
        self.charge['TS10'] = 0
        self.spin['TS10'] = 2

        self.geom['TS11'] = '''
        P         .217429     .000088    -.111249
        H         .246609    1.034668     .852164
        H         .262661   -1.025058     .861623
        H       -1.266418    -.010952    -.150626
        H       -2.504290     .000028     .105575
        '''
        self.charge['TS11'] = 0
        self.spin['TS11'] = 1

        self.geom['TS12'] = '''
        H         .000000     .000000    -.860287
        O         .000000     .000000     .329024
        H         .000000     .000000   -1.771905
        '''
        self.charge['TS12'] = 0
        self.spin['TS12'] = 2

        self.geom['TS13'] = '''
        H        1.262097    -.220097     .000000
        S         .000000     .223153     .000000
        H        -.500576   -1.115445     .000000
        H        -.761521   -2.234913     .000000
        '''
        self.charge['TS13'] = 0
        self.spin['TS13'] = 1

        self.geom['TS14'] = '''
        Cl        .018820    -.817301     .000000
        H        -.470488     .569480     .000000
        O         .018820    1.665579     .000000
        '''
        self.charge['TS14'] = 0
        self.spin['TS14'] = 2

        self.geom['TS15'] = '''
        C       -1.199577    -.011126    -.000030
        N        1.400715     .129862     .000015
        H       -1.426660    -.512932     .933057
        H       -1.419907    -.591382    -.888143
        H       -1.520237    1.022806    -.045783
        H         .188926     .126896     .001001
        H        1.570338    -.887667    -.000053
        '''
        self.charge['TS15'] = 0
        self.spin['TS15'] = 2

        self.geom['TS16'] = '''
        C       -1.394984   -0.449661    0.000703
        C       -0.435746    0.714063    0.002027
        N        1.927570   -0.378352    0.003036
        H       -1.200087   -1.120951   -0.835687
        H       -1.322095   -1.027884    0.921773
        H       -2.428713   -0.105352   -0.089334
        H       -0.417688    1.308482   -0.907201
        H       -0.441127    1.329095    0.897467
        H        0.828501    0.180593   -0.028561
        H        2.472592    0.498073    0.003910
        '''
        self.charge['TS16'] = 0
        self.spin['TS16'] = 2

        self.geom['TS17'] = '''
        C       -1.485700   -0.448156   -0.000019
        C       -0.505042    0.701740    0.000029
        N        1.865161   -0.340167   -0.000057
        H       -1.354193   -1.076505   -0.880503
        H       -1.354159   -1.076611    0.880385
        H       -2.517025   -0.086173    0.000025
        H       -0.522224    1.316118   -0.897218
        H       -0.522205    1.316029    0.897338
        H        0.665047    0.147961   -0.000034
        H        2.246644    0.159717   -0.804806
        H        2.246439    0.159133    0.805151
        '''
        self.charge['TS17'] = 0
        self.spin['TS17'] = 1

        self.geom['TS18'] = '''
        C       -1.260750   -0.000006    0.012291
        N        1.313255   -0.000005   -0.136782
        H       -1.583987    0.908538   -0.484744
        H       -1.463672   -0.004573    1.077302
        H       -1.584748   -0.903880   -0.492700
        H        0.043108   -0.000064   -0.151692
        H        1.480459    0.805577    0.467751
        H        1.480557   -0.805524    0.467808
        '''
        self.charge['TS18'] = 0
        self.spin['TS18'] = 1

        self.geom['TS19'] = '''
        C       -1.299623   -0.904853   -0.020155
        C       -1.205947    0.505817   -0.013414
        C        0.000000    1.183361    0.153301
        C        1.205948    0.505814   -0.013422
        C        1.299626   -0.904851   -0.020147
        H        2.168797   -1.327549   -0.515697
        H        1.032041   -1.454385    0.873166
        H        2.037130    1.085583   -0.398504
        H        0.000001    2.262913    0.085905
        H       -2.037133    1.085587   -0.398481
        H       -2.168796   -1.327540   -0.515716
        H       -0.000011   -1.181942   -0.520808
        H       -1.032059   -1.454394    0.873158
        '''
        self.charge['TS19'] = 0
        self.spin['TS19'] = 0

        self.geom['TS1'] = '''
        H        0.000480   -1.340627    0.000000
        Cl       0.000000    0.203252    0.000000
        H       -0.000480   -2.114659    0.000000
        '''
        self.charge['TS1'] = 0
        self.spin['TS1'] = 1

        self.geom['TS2'] = '''
        O        -.301064    -.108049    -.000008
        H        -.427945     .851569     .000016
        H        1.015486    -.100367     .000119
        H        1.820968     .113187    -.000073
        '''
        self.charge['TS2'] = 0
        self.spin['TS2'] = 1

        self.geom['TS3'] = '''
        C         .000000     .264813     .000000
        H        1.053429     .516668     .000000
        H        -.526627     .517025     .912250
        H        -.526627     .517025    -.912250
        H        -.000260   -1.117771     .000000
        H         .000084   -2.021825     .000000
        '''
        self.charge['TS3'] = 0
        self.spin['TS3'] = 1

        self.geom['TS4'] = '''
        C       -1.211487     .007968     .000407
        O        1.293965    -.108694     .000133
        H         .009476    -.118020     .002799
        H       -1.525529    -.233250    1.010070
        H       -1.430665    1.033233    -.278082
        H       -1.552710    -.710114    -.737702
        H        1.416636     .849894    -.000591
        '''
        self.charge['TS4'] = 0
        self.spin['TS4'] = 1

        self.geom['TS5'] = '''
        H        0.000000    0.000000    0.000000
        H        0.000000    0.000000    0.929474
        H        0.000000    0.000000   -0.929474
        '''
        self.charge['TS5'] = 0
        self.spin['TS5'] = 1

        self.geom['TS6'] = '''
        N       -1.150816    -.043932    -.102559
        O        1.179186    -.092696    -.010290
        H       -1.303185    -.547638     .766571
        H       -1.338913     .935808     .091854
        H        -.030687    -.153834    -.353184
        H        1.295009     .814753     .294991
        '''
        self.charge['TS6'] = 0
        self.spin['TS6'] = 1

        self.geom['TS7'] = '''
        C         .244117     .599916    1.702423
        H        -.675597     .278482    2.172939
        H         .351910    1.663786    1.537672
        H        1.140686     .065787    1.987822
        H         .057163     .139973     .397112
        Cl       -.137580    -.338090    -.959416
        '''
        self.charge['TS7'] = 0
        self.spin['TS7'] = 1

        self.geom['TS8'] = '''
        C        1.458334    -.446365     .025478
        C         .469423     .697422    -.027493
        O       -1.853037    -.314659    -.053055
        H        1.301764   -1.061079     .910737
        H        1.366585   -1.086189    -.851118
        H        2.482245    -.066879     .057150
        H         .471069    1.325443     .861037
        H         .533524    1.303495    -.928560
        H        -.630232     .207816    -.078465
        H       -2.267207     .388321     .465751
        '''
        self.charge['TS8'] = 0
        self.spin['TS8'] = 1

        self.geom['TS9'] = '''
        H         .146568   -1.128390     .000000
        F         .000000     .330422     .000000
        H        -.146568   -1.845410     .000000
        '''
        self.charge['TS9'] = 0
        self.spin['TS9'] = 1
        return None
    
    def calculate_bh(self, energies:dict, unit='kcal'):
        bhs = {}
        if unit == 'kcal':
            e_factor = 627.5095
        elif unit == 'kJ':
            e_factor = 2625.4996
        elif unit == 'eV':
            e_factor = 27.2114
        elif unit == 'au':
            e_factor = 1.0
        else:
            raise ValueError(f"Unit {unit} not recognized. Use 'au', 'kcal', 'kJ', or 'eV'.")
        # Reaction 1: H + HCl → H2 + Cl
        bhs['F1'] = (energies['TS1'] - energies['H'] - energies['HCl']) * e_factor
        bhs['R1'] = (energies['TS1'] - energies['H2'] - energies['Cl']) * e_factor
        # Reaction 2: OH + H2 → H + H2O
        bhs['F2'] = (energies['TS2'] - energies['OH'] - energies['H2']) * e_factor
        bhs['R2'] = (energies['TS2'] - energies['H'] - energies['H2O']) * e_factor
        # Reaction 3: CH3 + H2 → H + CH4
        bhs['F3'] = (energies['TS3'] - energies['CH3'] - energies['H2']) * e_factor
        bhs['R3'] = (energies['TS3'] - energies['H'] - energies['CH4']) * e_factor
        # Reaction 4: OH + CH4 → CH3 + H2O
        bhs['F4'] = (energies['TS4'] - energies['OH'] - energies['CH4']) * e_factor
        bhs['R4'] = (energies['TS4'] - energies['CH3'] - energies['H2O']) * e_factor
        # Reaction 5: H + H2 → H2 + H
        bhs['F5'] = (energies['TS5'] - energies['H'] - energies['H2']) * e_factor
        bhs['R5'] = (energies['TS5'] - energies['H2'] - energies['H']) * e_factor
        # Reaction 6: OH + NH3 → H2O + NH2
        bhs['F6'] = (energies['TS6'] - energies['OH'] - energies['NH3']) * e_factor
        bhs['R6'] = (energies['TS6'] - energies['H2O'] - energies['NH2']) * e_factor
        # Reaction 7: HCl + CH3 → Cl + CH4
        bhs['F7'] = (energies['TS7'] - energies['HCl'] - energies['CH3']) * e_factor
        bhs['R7'] = (energies['TS7'] - energies['Cl'] - energies['CH4']) * e_factor
        # Reaction 8: OH + C2H6 → H2O + C2H5
        bhs['F8'] = (energies['TS8'] - energies['OH'] - energies['C2H6']) * e_factor
        bhs['R8'] = (energies['TS8'] - energies['H2O'] - energies['C2H5']) * e_factor
        # Reaction 9: F + H2 → HF + H
        bhs['F9'] = (energies['TS9'] - energies['F'] - energies['H2']) * e_factor
        bhs['R9'] = (energies['TS9'] - energies['HF'] - energies['H']) * e_factor
        # Reaction 10: O + CH4 → OH + CH3
        bhs['F10'] = (energies['TS10'] - energies['O'] - energies['CH4']) * e_factor
        bhs['R10'] = (energies['TS10'] - energies['OH'] - energies['CH3']) * e_factor
        # Reaction 11: H + PH3 → PH2 + H2
        bhs['F11'] = (energies['TS11'] - energies['H'] - energies['PH3']) * e_factor
        bhs['R11'] = (energies['TS11'] - energies['PH2'] - energies['H2']) * e_factor
        # Reaction 12: H + OH → H2 + O
        bhs['F12'] = (energies['TS12'] - energies['H'] - energies['OH']) * e_factor
        bhs['R12'] = (energies['TS12'] - energies['H2'] - energies['O']) * e_factor
        # Reaction 13: H + H2S → H2 + HS
        bhs['F13'] = (energies['TS13'] - energies['H'] - energies['H2S']) * e_factor
        bhs['R13'] = (energies['TS13'] - energies['H2'] - energies['HS']) * e_factor
        # Reaction 14: O + HCl → OH + Cl
        bhs['F14'] = (energies['TS14'] - energies['O'] - energies['HCl']) * e_factor
        bhs['R14'] = (energies['TS14'] - energies['OH'] - energies['Cl']) * e_factor
        # Reaction 15: NH2 + CH3 → CH4 + NH
        bhs['F15'] = (energies['TS15'] - energies['NH2'] - energies['CH3']) * e_factor
        bhs['R15'] = (energies['TS15'] - energies['CH4'] - energies['NH']) * e_factor
        # Reaction 16: NH2 + C2H5 → C2H6 + NH
        bhs['F16'] = (energies['TS16'] - energies['NH2'] - energies['C2H5']) * e_factor
        bhs['R16'] = (energies['TS16'] - energies['C2H6'] - energies['NH']) * e_factor
        # Reaction 17: NH2 + C2H6 → C2H5 + NH3
        bhs['F17'] = (energies['TS17'] - energies['NH2'] - energies['C2H6']) * e_factor
        bhs['R17'] = (energies['TS17'] - energies['C2H5'] - energies['NH3']) * e_factor
        # Reaction 18: NH2 + CH4 → CH3 + NH3
        bhs['F18'] = (energies['TS18'] - energies['NH2'] - energies['CH4']) * e_factor
        bhs['R18'] = (energies['TS18'] - energies['CH3'] - energies['NH3']) * e_factor
        # Reaction 19: C5H8 → C5H8
        bhs['F19'] = (energies['TS19'] - energies['C5H8']) * e_factor
        bhs['R19'] = (energies['TS19'] - energies['C5H8']) * e_factor
        return bhs
    
    def get_reference(self, unit='kcal'):
        bh_ref = {}
        e_factor = 1.0
        if unit == 'kcal':
            e_factor = 1.0
        elif unit == 'kJ':
            e_factor = 4.184
        elif unit == 'eV':
            e_factor = 0.0433641
        elif unit == 'au':
            e_factor = 0.00159362
        else:
            raise ValueError(f"Unit {unit} not recognized. Use 'au', 'kcal', 'kJ', or 'eV'.")
        # Reference barrier heights from the original paper (kcal/mol)
        bh_ref['F1'] = 5.7
        bh_ref['R1'] = 8.7
        bh_ref['F2'] = 5.1
        bh_ref['R2'] = 21.2
        bh_ref['F3'] = 12.1
        bh_ref['R3'] = 15.3
        bh_ref['F4'] = 6.7
        bh_ref['R4'] = 19.6
        bh_ref['F5'] = 9.6
        bh_ref['R5'] = 9.6
        bh_ref['F6'] = 3.2
        bh_ref['R6'] = 12.7
        bh_ref['F7'] = 1.7
        bh_ref['R7'] = 7.9
        bh_ref['F8'] = 3.4
        bh_ref['R8'] = 19.9
        bh_ref['F9'] = 1.8
        bh_ref['R9'] = 33.4
        bh_ref['F10'] = 13.7
        bh_ref['R10'] = 8.1
        bh_ref['F11'] = 3.1
        bh_ref['R11'] = 23.2
        bh_ref['F12'] = 10.7
        bh_ref['R12'] = 13.1
        bh_ref['F13'] = 3.5
        bh_ref['R13'] = 17.3
        bh_ref['F14'] = 9.8
        bh_ref['R14'] = 10.4
        bh_ref['F15'] = 8.0
        bh_ref['R15'] = 22.4
        bh_ref['F16'] = 7.5
        bh_ref['R16'] = 18.3
        bh_ref['F17'] = 10.4
        bh_ref['R17'] = 17.4
        bh_ref['F18'] = 14.5
        bh_ref['R18'] = 17.8
        bh_ref['F19'] = 38.4
        bh_ref['R19'] = 38.4
        for key in bh_ref:
            bh_ref[key] = bh_ref[key] * e_factor
        return bh_ref
    


    

def check():
    geom = HTBH38()
    names = geom.get_names()
    print('unit', geom.unit)
    print('number of molecules:', len(names))
    energy={}
    for name in names:
        print('name:', name)
        mol = geom.get_molecule(name, basis='sto-3g')
        print(mol.natm)
        print(mol.nao)
        energy[name] = 0.01
    bhs = geom.calculate_bh(energy, unit='kcal')
    bh_ref = geom.get_reference(unit='kcal')
    for i, (ref, calc) in enumerate(zip(bh_ref.items(), bhs.items())):
        print(f"{i+1:2d} {ref[0]:10s} {ref[1]:10.4f} {calc[1]:10.4f} {ref[1]-calc[1]:10.4f}")
    return None


if __name__ == '__main__':
    check()
    pass