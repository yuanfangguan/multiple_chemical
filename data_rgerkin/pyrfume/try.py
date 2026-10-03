import sys
sys.path.append('/local/disk3/gyuanfan/Olfactory_2025/data_rgerkin/pyrfume')

import pyrfume

import pyrfume
behavior = pyrfume.load_data('snitz_2013/behavior.csv')
molecules = pyrfume.load_data('snitz_2013/molecules.csv')
#print(behavior)
print(molecules)
print(molecules.columns)

behavior = pyrfume.load_data('bushdid_2014/behavior.csv')
molecules = pyrfume.load_data('bushdid_2014/molecules.csv')
stimuli = pyrfume.load_data('bushdid_2014/stimuli.csv')
print(behavior.columns)
print(molecules)
print(molecules.columns)

