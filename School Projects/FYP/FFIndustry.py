
# Fama French 5 Industry Definition with SIC codes: 
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_5_ind_port.html
# Fama French 12 Industry Definition with SIC codes:
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_12_ind_port.html

### FF5 Industry
Consumer=[(100,999),
          (2000,2399),
          (2700,2749),
          (2770,2799),
          (3100,3199),
          (3940,3989),
          (2500,2519),
          (2590,2599),
          (3630,3659),
          (3710,3711),
          (3714,3714),
          (3716,3716),
          (3750,3751),
          (3792,3792),
          (3900,3939),
          (3990,3999),
          (5000,5999),
          (7200,7299),
          (7600,7699)]

Manufacturing=[(2520,2589),
              (2600,2699),
              (2750,2769),
              (2800,2829),
              (2840,2899),
              (3000,3099),
              (3200,3569),
              (3580,3621),
              (3623,3629),
              (3700,3709),
              (3712,3713),
              (3715,3715),
              (3717,3749),
              (3752,3791),
              (3793,3799),
              (3860,3899),
              (1200,1399),
              (2900,2999),
              (4900,4949)]

HiTec =  [(3570,3579),
          (3622,3622), 
          (3660,3692),
          (3694,3699),
          (3810,3839),
          (7370,7372),  
          (7373,7373),  
          (7374,7374),    
          (7375,7375), 
          (7376,7376),  
          (7377,7377),  
          (7378,7378),  
          (7379,7379),  
          (7391,7391), 
          (8730,8734), 
          (4800,4899)]

Healthcare = [(2830,2839),
              (3693,3693),
              (3840,3859),
              (8000,8099)]

FF5_list = {'Consumer':Consumer,
            'Manufacturing':Manufacturing,
            'HiTec':HiTec,
            'Healthcare':Healthcare}

### FF12 Industry
Cons_Nondurables = [(100,999),
                  (2000,2399),
                  (2700,2749),
                  (2770,2799),
                  (3100,3199),
                  (3940,3989)]

Cons_Durables = [(2500,2519),
                  (2590,2599),
                  (3630,3659),
                  (3710,3711),
                  (3714,3714),
                  (3716,3716),
                  (3750,3751),
                  (3792,3792),
                  (3900,3939),
                  (3990,3999)]

Manufacturing = [(2520,2589),
                  (2600,2699),
                  (2750,2769),
                  (3000,3099),
                  (3200,3569),
                  (3580,3629),
                  (3700,3709),
                  (3712,3713),
                  (3715,3715),
                  (3717,3749),
                  (3752,3791),
                  (3793,3799),
                  (3830,3839),
                  (3860,3899)]

Energy = [(1200,1399),
          (2900,2999)]

Chemicals = [(2800,2829),
              (2840,2899)]

Bus_Equipment = [(3570,3579),
                  (3660,3692),
                  (3694,3699),
                  (3810,3829),
                  (7370,7379)]
Telecom = [(4800,4899)]

Utilities = [(4900,4949)]

Shops = [(5000,5999),
          (7200,7299),
          (7600,7699)]

Healthcare = [(2830,2839),
              (3693,3693),
              (3840,3859),
              (8000,8099)]

Finance = [(6000,6999)]

FF12_list = {'Cons_Nondurables':Cons_Nondurables,
            'Cons_Durables':Cons_Durables,
            'Manufacturing':Manufacturing,
            'Energy':Energy,
            'Chemicals':Chemicals,
            'Bus_Equipment':Bus_Equipment,
            'Telecom':Telecom,
            'Utilities':Utilities,
            'Shops':Shops,
            'Healthcare':Healthcare,
            'Finance':Finance}

### BEA
# Computers and peripheral equipment
Comp_Per_Eq = [(3570,3579),
               (3680,3689),
               (3695)]
# Software
Software = [(7372)]

# Pharmaceuticals
Pharma = [(2830), 
          (2831),
          (2833,2836)]
# Semiconductor
Semiconductor = [(3661,3666),
                 (3669,3679)]
# Aerospace product and parts
Aerospace = [(3720),
             (3721),
             (3724),
             (3728),
             (3760)]
# Communication equipment
Comm_Eq = [(3576),
           (3661),
           (3663),
           (3669),
           (3679)]
# Computer system design
Comp_Sys_Des = [(7370), 
                (7371),
                (7373)]
# Motor vehicles, bodies and trailers, and parts
Motors = [(3585),
          (3711),
          (3713,3716)]
# Navigational, measuring, electromedical, and control instruments
Control_Inst = [(3812),
                (3822),
                (3823),
                (3825),
                (3826),
                (3829),
                (3842),
                (3844),
                (3845)]
# Scientific research and development
Sci_Rnd = [(8731)]

BEA_SIC = {'Comp_Per_Eq':Comp_Per_Eq,
            'Software':Software,
            'Pharma':Pharma,
            'Semiconductor':Semiconductor,
            'Aerospace':Aerospace,
            'Comm_Eq':Comm_Eq,
            'Comp_Sys_Des':Comp_Sys_Des,
            'Motors':Motors,
            'Control_Inst':Control_Inst,
            'Sci_Rnd':Sci_Rnd}

BEA_rnd_dep = {'Comp_Per_Eq':0.363,
               'Software':0.308,
               'Pharma':0.112,
               'Semiconductor':0.226,
               'Aerospace':0.339,
               'Comm_Eq':0.192,
               'Comp_Sys_Des':0.489,
               'Motors':0.733,
               'Control_Inst':0.329,
               'Sci_Rnd':0.295}

def broadcast(x, lower, upper, value):
    if type(x) == int:
        if lower <= x <= upper:
            return value
        else:
            return x
    else:
        return x 


