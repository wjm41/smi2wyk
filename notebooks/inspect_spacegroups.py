#%%
import pandas as pd
from smi2wyk.utils import spacegroup_from_crystal
from smi2wyk.wren_code.utils import count_wyks, count_params, count_distinct_wyckoff_letters, return_spacegroup_number

import pandas as pd
from tqdm import tqdm

df_organic = pd.read_csv("csd_organic.csv").head(10000)

tqdm.pandas()
# df_organic['n_atoms'] = df_organic['wyckoff'].progress_apply(count_wyks)
# df_organic['n_wyk'] = df_organic['wyckoff'].progress_apply(count_distinct_wyckoff_letters)
df_organic['spg'] = df_organic['wyckoff'].progress_apply(return_spacegroup_number)
df_organic['spg_csd'] = df_organic['identifier'].progress_apply(spacegroup_from_crystal)
# df_organic['n_param'] = df_organic['wyckoff'].progress_apply(count_params)
print(df_organic.query('spg != spg_csd'))
# %%
