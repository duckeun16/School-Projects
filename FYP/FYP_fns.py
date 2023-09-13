import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.stats.mstats import winsorize
from scipy.stats import skew, kurtosis
from scipy import stats

def get_age(reg_df):
    # age within sample period using fyear - min(fyear)
    data_starts = reg_df.groupby('gvkey')['fyear'].min()
    reg_df['age'] = reg_df.apply(lambda x: x['fyear'] - data_starts.loc[x['gvkey']] + 1 \
                                 if x['gvkey'] in data_starts.index else None, axis=1)
    
    # age using ipodate, datadate - ipodate
    # replace ipodate missing values with 0
    reg_df['ipodate'] = np.where(reg_df['ipodate'].isnull() == True, 0, reg_df['ipodate'].values)

    datadate = reg_df['datadate'].apply(pd.to_datetime)
    ipodate = reg_df['ipodate'].apply(pd.to_datetime)

    conditions = [datadate > ipodate, 
                  datadate < ipodate,
                  datadate == ipodate]
    choices = [((datadate - ipodate).dt.days / 365).apply(np.floor), 0, 0]

    reg_df['age_ipo'] = np.select(conditions, choices, 0)

    # age is maximum of age_ipo and age
    reg_df['age'] = reg_df[['age','age_ipo']].apply(np.max, axis=1)
    reg_df = reg_df.drop(columns=['datadate','ipodate','age_ipo'])
    return reg_df

def missing_to_0(reg_df, varlist):
    for var in varlist:
        reg_df[var] = reg_df[var].apply(lambda x: 0 if np.isnan(x) else x)
    return reg_df # not need to return, actually makes changes to reg_df directly
        
def missing_to_mean(reg_df, impt_vars):
    reg_keys = list(reg_df['gvkey'].unique())
    before = reg_df['gvkey'].nunique()
    print('before', before)
    for key in tqdm(reg_keys):
        for var in impt_vars:
            key_df = reg_df.loc[reg_df['gvkey'] == key, var]
            key_df = key_df.fillna(key_df.mean())
            reg_df.loc[reg_df['gvkey'] == key, var] = key_df
            
    # drop firms that do not have any obs per important vars to calculate mean
    bools = reg_df[impt_vars].progress_apply(lambda x: False if x.isnull().sum() > 0 else True, axis=1)
    print('dropped firms:', reg_df[~bools]['gvkey'].nunique())
    reg_df = reg_df[bools]
    print('left:', reg_df['gvkey'].nunique())
    
#     # fill missing values by firm's mean
#     for var in impt_vars:
#         reg_df[var] = reg_df.groupby('gvkey')[var].transform('mean')
        
#     # Define a function to drop any group (i.e., firm) that has missing values in the specified columns
#     def drop_missing_firms(df):
#         if df[impt_vars].isnull().any().any(): # for each firm (group of obs), if any missing rows in any missing columns, drop
#             return None
#         else:
#             return df
        
#     # Apply the function to each group in the dataframe
#     reg_df = reg_df.groupby('gvkey').apply(drop_missing_firms)
#     reg_df = reg_df.droplevel(1)
#     reg_df = reg_df.reset_index(drop=True)
    
#     left = reg_df['gvkey'].nunique()
#     print('dropped:', before - left)
#     print('left:', left)
    return reg_df

def missing_to_ffill(reg_df, impt_vars):
    reg_keys = list(reg_df['gvkey'].unique())
    print('before',reg_df['gvkey'].nunique())
    for key in tqdm(reg_keys):
        for var in impt_vars:
            key_df = reg_df.loc[reg_df['gvkey'] == key, var]
            key_df = key_df.fillna(method='ffill')
            reg_df.loc[reg_df['gvkey'] == key, var] = key_df
            
    # drop firms that do not have any obs per important vars to calculate mean
    bools = reg_df[impt_vars].progress_apply(lambda x: False if x.isnull().sum() > 0 else True, axis=1)
    print('dropped firms:', reg_df[~bools]['gvkey'].nunique())
    reg_df = reg_df[bools]
    print('left:', reg_df['gvkey'].nunique())
    return reg_df

def missing_to_bfill(reg_df, impt_vars):
    reg_keys = list(reg_df['gvkey'].unique())
    print('before',reg_df['gvkey'].nunique())
    for key in tqdm(reg_keys):
        for var in impt_vars:
            key_df = reg_df.loc[reg_df['gvkey'] == key, var]
            key_df = key_df.fillna(method='bfill')
            reg_df.loc[reg_df['gvkey'] == key, var] = key_df
            
    # drop firms that do not have any obs per important vars to calculate mean
    bools = reg_df[impt_vars].progress_apply(lambda x: False if x.isnull().sum() > 0 else True, axis=1)
    print('dropped firms:', reg_df[~bools]['gvkey'].nunique())
    reg_df = reg_df[bools]
    print('left:', reg_df['gvkey'].nunique())
    return reg_df

# Convert objects into floats
def to_float(reg_df, varlist):
    for var in varlist:
        reg_df[var] = reg_df[var].apply(lambda x: float(x))
    return reg_df # not need to return, actually makes changes to reg_df directly

def missing_neg_plot(reg_df, w=12, h=6,neg_check=True, horizontal = False):
    if neg_check == True:
        r = 3
        c = 1
        if horizontal == True:
            r = 1
            c = 3
        fig, ax = plt.subplots(r,c, figsize=(w,h))
        # percentage of missing data
        df_missing_pct = reg_df.isnull().sum()/len(reg_df) * 100
        ax[0].set_title('% of data missing')
        ax[0].bar(df_missing_pct.index, df_missing_pct)
        ax[0].tick_params(axis="x",rotation=90,labelsize=15)
        plt.tight_layout()

        # percentage of data == 0
        df_zero_pct = (reg_df == 0).sum()/len(reg_df) * 100
        ax[1].set_title('% of data == 0')
        ax[1].bar(df_zero_pct.index, df_zero_pct, color='tab:orange')
        ax[1].tick_params(axis="x",rotation=90,labelsize=15)
        plt.tight_layout()

        # percentage of data < 0
        reg_df = reg_df.select_dtypes(include=['int64','float64']) # only number columns
        df_neg_pct = (reg_df < 0).sum()/len(reg_df) * 100
        ax[2].set_title('% of data < 0')
        ax[2].bar(df_neg_pct.index, df_neg_pct, color='tab:red')
        ax[2].tick_params(axis="x",rotation=90,labelsize=15)
        plt.tight_layout()

    else:
        r = 2
        c = 1
        if horizontal == True:
            r = 1
            c = 2
        fig, ax = plt.subplots(r,c, figsize=(w,h))
        # percentage of missing data
        df_missing_pct = reg_df.isnull().sum()/len(reg_df) * 100
        ax[0].set_title('% of data missing')
        ax[0].bar(df_missing_pct.index, df_missing_pct)
        ax[0].tick_params(axis="x",rotation=90,labelsize=15)
        plt.tight_layout()

        # percentage of data == 0
        df_zero_pct = (reg_df == 0).sum()/len(reg_df) * 100
        ax[1].set_title('% of data == 0')
        ax[1].bar(df_zero_pct.index, df_zero_pct, color='tab:orange')
        ax[1].tick_params(axis="x",rotation=90,labelsize=15)
        plt.tight_layout()
    return None

def Industry_dummies(reg_df, base):
    FF_industry = list(reg_df['sic'].unique())
    
    for industry in FF_industry:
        reg_df[industry] = None
        reg_df.loc[reg_df['sic'] == industry,industry] = 1
        reg_df.loc[reg_df['sic'] != industry,industry] = 0
        reg_df[industry] = reg_df[industry].apply(lambda x: int(x))
    # drop base column to avoid dummy variable trap
    reg_df = reg_df.drop(columns=[base])
    return reg_df

def get_know_cap(reg_df):
    # computing knowledge capital of internally generated I/A 
    reg_df['knowledge_cap'] = None
    tickers = list(reg_df['gvkey'].unique())
    len_tkr_data = []

    for tkr in tickers:
        each_stock = reg_df['gvkey'] == tkr
        # get R&D dep rate
        dep_factor = (1 - reg_df.loc[each_stock,'rnd_dep'].iloc[0])

        # x is length of R&D expense data for each tickers
        xrd = reg_df.loc[each_stock,'xrd']
        x = len(xrd)

        # create x by x matrix with lower triangle = 1, upper = 0
        mat = np.tri(x, x, 0) 
        # array list with length of R&D expense starting from 1
        arr = [np.arange(1,x+1)]
        # create R&D depreication matrix starting with powers
        # repeat array list into matrix and transpose,
        # subtract array list to create lagging data counts in each columns
        # censor out upper triangle with mat
        rnd_dep_mat = (np.repeat(arr, x, axis = 0).T - arr) * mat
        # map out depreciation rate on the matrix to create actual multipliers to each yr's R&D expense
        rnd_dep_mat = dep_factor ** (rnd_dep_mat) * mat
        # map out R&D expense to R&D dep rate matrix
        G_mat = rnd_dep_mat * xrd.values
        # sum matrix over the columns
        G_it = G_mat.sum(axis=1)

        G_it = pd.Series(G_it)
        # Set initial stock = 0
        G_it.iloc[0] = 0
        G_it.index = reg_df.loc[each_stock,'knowledge_cap'].index
        reg_df.loc[each_stock,'knowledge_cap'] = G_it
    return reg_df

def get_organ_cap(reg_df):
    # computing organization capital of internally generated I/A
    reg_df['organization_cap'] = None
    tickers = list(reg_df['gvkey'].unique())
    len_tkr_data = []
    # SG&A dep rate is constant over tickers
    sga_dep = 0.2
    dep_factor = (1 - sga_dep)

    for tkr in tickers:
        each_stock = reg_df['gvkey'] == tkr

        # only 30% of xsga is counted as intangible investment
        xsga = reg_df.loc[each_stock,'xsga'] * 0.3 
        x = len(xsga)

        # create x by x matrix with lower triangle = 1, upper = 0
        mat = np.tri(x, x, 0) 
        # array list with length of SG&A expense starting from 1
        arr = [np.arange(1,x+1)]
        # create SG&A depreication matrix starting with powers
        sga_dep_mat = (np.repeat(arr, x, axis = 0).T - arr) * mat
        # map out depreciation rate on the matrix to create actual multipliers to each yr's R&D expense
        sga_dep_mat = dep_factor ** (sga_dep_mat) * mat
        # map out R&D expense to R&D dep rate matrix
        O_mat = sga_dep_mat * xsga.values
        # sum matrix over the columns
        O_it = O_mat.sum(axis=1)

        O_it = pd.Series(O_it)
        # Set initial stock = 0
        O_it.iloc[0] = 0
        O_it.index = reg_df.loc[each_stock,'organization_cap'].index
        reg_df.loc[each_stock,'organization_cap'] = O_it
    return reg_df

def get_reg_vars(reg_df):
    reg_df.index = reg_df['gvkey']
    reg_df = reg_df.drop(columns=['gvkey'])

    # Total Debt
    reg_df['T_Debt'] = reg_df['dlc'] + reg_df['dltt']
    
    ## Calculate Market Value of Common Equity
    reg_df['MV_CE'] = (reg_df['prcc_f'] * reg_df['csho'])
#     reg_df['ME'] = (reg_df['prcc_c'] * reg_df['csho'])
    reg_df = reg_df.drop(columns=['prcc_f','csho'])
    
    ## Calculate Market Value of Firm
    # Fair value of the firm
    reg_df['FMV'] = (reg_df['at'] - reg_df['ceq'] + reg_df['MV_CE'])
    # Firm's Market Value (Method Paper)
#     reg_df['MV_firm'] = reg_df['MV_CE'] + reg_df['T_Debt'] - reg_df['act']

    ### Dependent Variables:
#     ## (Book) Leverage using only LTD
#     reg_df['LEV_LTD'] = reg_df['dltt'] / reg_df['at'] # Main Paper
#     reg_df[['dltt','LEV_LTD']] = reg_df[['dltt','LEV_LTD']].apply(lambda x: 0 if x['dltt'] == 0 else x, axis=1)
    
    reg_df['LEV_TD'] = reg_df['T_Debt'] / reg_df['at'] # other literature (USED)
    # reg_df[['T_Debt','LEV_TD']] = reg_df[['T_Debt','LEV_TD']].apply(lambda x: 0 if x['T_Debt'] == 0 else x, axis=1)
    
    ## Market Leverage using only LTD
#     reg_df['Mkt_LEV_firm'] = reg_df['dltt'] / reg_df['MV_firm'] # Method Paper MV
#     reg_df[['dltt','Mkt_LEV_firm']] = reg_df[['dltt','Mkt_LEV_firm']].apply(lambda x: 0 if x['dltt'] == 0 else x, axis=1)
    
#     reg_df['Mkt_LEV_CE'] = reg_df['dltt'] / reg_df['MV_CE'] # Main Paper MV
#     reg_df[['dltt','Mkt_LEV_CE']] = reg_df[['dltt','Mkt_LEV_CE']].apply(lambda x: 0 if x['dltt'] == 0 else x, axis=1)
    
#     reg_df['Mkt_LEV_LTD'] = reg_df['dltt'] / reg_df['FMV'] # Other literature
#     reg_df[['dltt','Mkt_LEV_LTD']] = reg_df[['dltt','Mkt_LEV_LTD']].apply(lambda x: 0 if x['dltt'] == 0 else x, axis=1)
    
    reg_df['Mkt_LEV'] = reg_df['T_Debt'] / reg_df['FMV'] # Other literature (USED)
    # reg_df[['T_Debt','Mkt_LEV']] = reg_df[['T_Debt','Mkt_LEV']].apply(lambda x: 0 if x['T_Debt'] == 0 else x, axis=1)
    
    # Market_to_Book of common equity
    reg_df['Market_to_Book'] = reg_df['MV_CE'] / reg_df['ceq']
    # reg_df[['MV_CE','Market_to_Book']] = reg_df[['MV_CE','Market_to_Book']].apply(lambda x: 0 if x['MV_CE'] == 0 else x, axis=1)
    
    # Tobin_Q
#     reg_df['Tobin_Q'] = reg_df['FMV'] / reg_df['at']
#     reg_df[['FMV','at']] = reg_df[['FMV','at']].apply(lambda x: 0 if x['FMV'] == 0 else x, axis=1)
    
#     # Altman_Z = 3.3*(EBIT/AT) +0.99*(SALE/AT) +0.6*(ME/LT) +1.2*(ACT/AT) +1.4*(RE/AT)
#     reg_df['EBIT_AT'] = (reg_df['ebit'] / reg_df['at'])
#     reg_df[['ebit','EBIT_AT']] = reg_df[['ebit','EBIT_AT']].apply(lambda x: 0 if x['ebit'] == 0 else x, axis=1)
#     reg_df['SALE_AT'] = (reg_df['sale'] / reg_df['at'])
#     reg_df[['sale','SALE_AT']] = reg_df[['sale','SALE_AT']].apply(lambda x: 0 if x['sale'] == 0 else x, axis=1)
#     reg_df['MVCE_LT'] = (reg_df['MV_CE'] / reg_df['lt'])
#     reg_df[['MV_CE','MVCE_LT']] = reg_df[['MV_CE','MVCE_LT']].apply(lambda x: 0 if x['MV_CE'] == 0 else x, axis=1)
#     reg_df['ACT_AT'] = (reg_df['act'] / reg_df['at'])
#     reg_df[['act','ACT_AT']] = reg_df[['act','ACT_AT']].apply(lambda x: 0 if x['act'] == 0 else x, axis=1)
#     reg_df['RE_AT'] = (reg_df['re'] / reg_df['at'])
#     reg_df[['re','RE_AT']] = reg_df[['re','RE_AT']].apply(lambda x: 0 if x['re'] == 0 else x, axis=1)
    
#     reg_df['Altman_Z'] = 3.3 * reg_df['EBIT_AT'] \
#                        + 0.99 * reg_df['SALE_AT'] \
#                        + 0.6 * reg_df['MVCE_LT'] \
#                        + 1.2 * reg_df['ACT_AT'] \
#                        + 1.4 * reg_df['RE_AT']
    
#     reg_df = reg_df.drop(columns=['ebit','lt','re','act','EBIT_AT','SALE_AT','MVCE_LT','ACT_AT','RE_AT'])
    
    # Operating profitability
    reg_df['Op_profit'] = reg_df['ebitda'] / reg_df['at']
    reg_df[['ebitda','Op_profit']] = reg_df[['ebitda','Op_profit']].apply(lambda x: 0 if x['ebitda'] == 0 else x, axis=1)
#     # Net Profitability
#     reg_df['Net_Profit_Margin'] = reg_df['ni'] / reg_df['sale']
#     reg_df[['ni','Net_Profit_Margin']] = reg_df[['ni','Net_Profit_Margin']].apply(lambda x: 0 if x['ni'] == 0 else x, axis=1)
#     # Tax Burden
#     reg_df['Tax_Burden'] = 1 - (reg_df['pi'] / reg_df['ni'])
#     reg_df[['pi','Tax_Burden']] = reg_df[['pi','Tax_Burden']].apply(lambda x: 0 if x['pi'] == 0 else x, axis=1)

    # Cash
    reg_df['Cash'] = reg_df['ch'] + reg_df['che']
    # Cash liquidity
    reg_df['Cash_liq'] = reg_df['Cash'] / reg_df['at']
    # reg_df[['Cash','Cash_liq']] = reg_df[['Cash','Cash_liq']].apply(lambda x: 0 if x['Cash'] == 0 else x, axis=1)

    # Log Total Asset
    reg_df['log_asset'] = reg_df['at'].apply(lambda x: np.log(x))
    # Log Market Capitalization
    reg_df['log_MV_CE'] = reg_df['MV_CE'].apply(lambda x: np.log(x))
    # Log Sales
    reg_df['log_sale'] = reg_df['sale'].apply(lambda x: np.log(x))

    # Intan_cap/Asset
    reg_df['intan_cap_AT'] = reg_df['intan_cap'] / reg_df['at']   
#     reg_df[['intan_cap','intan_cap_AT']] = reg_df[['intan_cap','intan_cap_AT']]\
#     .apply(lambda x: 0 if x['intan_cap'] == 0 else x, axis=1)

    # Intan/Asset
    reg_df['intan_AT'] = reg_df['intan'] / reg_df['at']
#     reg_df[['intan','intan_AT']] = reg_df[['intan','intan_AT']].apply(lambda x: 0 if x['intan'] == 0 else x, axis=1)

    # Intan_cap(Less G/W)/Asset
    reg_df['intan_cap_lessgdwl_AT'] = reg_df['intan_cap_lessgdwl'] / reg_df['at']
#     reg_df[['intan_cap_lessgdwl','intan_cap_lessgdwl_AT']] = reg_df[['intan_cap_lessgdwl','intan_cap_lessgdwl_AT']]\
#     .apply(lambda x: 0 if x['intan_cap_lessgdwl'] == 0 else x, axis=1)

    # Intan(Less G/W)/Asset
    reg_df['intan_lessgdwl_AT'] = reg_df['intan_lessgdwl'] / reg_df['at']
#     reg_df[['intan_lessgdwl','intan_lessgdwl_AT']] = reg_df[['intan_lessgdwl','intan_lessgdwl_AT']]\
#     .apply(lambda x: 0 if x['intan_lessgdwl'] == 0 else x, axis=1)

    # Knowledge_cap/Asset
    reg_df['know_cap_AT'] = reg_df['knowledge_cap'] / reg_df['at']
    print(reg_df['knowledge_cap'].shape,reg_df['know_cap_AT'].shape)
#     reg_df['know_cap_AT'].apply(lambda x: 0 if np.isnan(x) == True else x)
    #reg_df[['knowledge_cap','know_cap_AT']] = reg_df[['knowledge_cap','know_cap_AT']]\
    #.apply(lambda x: 0 if x['knowledge_cap'] == 0 else x, axis=1)
    
    # Organization_cap/Asset
    reg_df['organ_cap_AT'] = reg_df['organization_cap'] / reg_df['at']
    print(reg_df['organization_cap'].shape,reg_df['organ_cap_AT'].shape)
#     reg_df['organ_cap_AT'].apply(lambda x: 0 if np.isnan(x) == True else x)
    #reg_df[['organization_cap','organ_cap_AT']] = reg_df[['organization_cap','organ_cap_AT']]\
    #.apply(lambda x: 0 if x['organization_cap'] == 0 else x, axis=1)

    # NetPPE/Asset (Tangibility) --> use as main
    reg_df['PPENT_AT'] = reg_df['ppent'] / reg_df['at']
#     reg_df[['ppent','PPENT_AT']] = reg_df[['ppent','PPENT_AT']].apply(lambda x: 0 if x['ppent'] == 0 else x, axis=1)
    # # GrossPPE/Asset (Tangibility)
    # reg_df['PPEGT_AT'] = reg_df['ppegt'] / reg_df['at']

    # Drop unnecessary columns
    reg_df = reg_df.drop(columns=['ch','che','ebitda','Cash','dlc','dltt']) #'ni','pi', 'ceq', ,'FMV'
    
    reg_df = reg_df.reset_index()
    return reg_df

def find_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5*IQR)) | (df > (q3 + 1.5*IQR)))]
    return outliers

def vis_outliers(reg_df):
    # Percentage of outliers
    outliers_df = find_outliers_IQR(reg_df).notnull().sum() / reg_df.count() * 100 # reg_df.notnull().sum()
    outliers_df.plot(kind='bar');
    return None

# winsorized df
def get_winsor_df(reg_df, winsor1, winsor2, pct1=0.05, pct2=0.1):
    reg_df = reg_df.copy()
# ### VER 1
#     winsor_less = ['at', 'gdwl', 'ppent', 'sale',
#                    'knowledge_cap', 'organization_cap',
#                    'intan_cap', 'intan_cap_lessgdwl', 
#                    'intan', 'intan_lessgdwl',
#                    'T_Debt', 'MV_CE', 'LEV_LTD', 'LEV_TD', 'Mkt_LEV_CE', 'Mkt_LEV', 
#                    'log_asset', 'log_MV_CE', 'log_sale', 'Cash_liq',
#                    'intan_AT', 'intan_lessgdwl_AT', 'intan_cap_AT', 'intan_cap_lessgdwl_AT',
#                    'know_cap_AT', 'organ_cap_AT', 'PPENT_AT']
#     winsor_more = ['Market_to_Book', 'Altman_Z', 'Op_profit', 'log_sale']

## VER 2: winsor estimates
    industry_dummies = list(reg_df['sic'].unique())
    industry_dummies.remove('Other')

#     winsor_less = ['gdwl', 'ppent', 
#                    'T_Debt', 'MV_CE', 'LEV_TD', 'Mkt_LEV', 
#                    'log_MV_CE', 'Cash_liq',
#                    'intan', 'intan_lessgdwl',
#                    'intan_AT', 'intan_lessgdwl_AT',
#                    'at','log_asset', 'sale', 'log_sale',
#                    'PPENT_AT'] 
#                     #'intan', 'intan_lessgdwl',
#                     #'intan_AT', 'intan_lessgdwl_AT', 
#                     # 'LEV_LTD', 'Mkt_LEV_CE',
#     # winsorize estimates and those with many outliers
#     winsor_more = ['Market_to_Book', 'Op_profit', 
#                    'knowledge_cap', 'organization_cap', 'intan_cap', 'intan_cap_lessgdwl',
#                    'know_cap_AT', 'organ_cap_AT', 'intan_cap_AT', 'intan_cap_lessgdwl_AT'] # 'Altman_Z', 

#     winsor_cols = ['gvkey', 'fyear','sic',
#                    'at', 'gdwl', 'ppent', 'sale', 'age', 'MTR_BI', 'MTR_AI',
#                    'knowledge_cap', 'organization_cap', 
#                    'intan_cap', 'intan_cap_lessgdwl', 
#                    'intan', 'intan_lessgdwl',
#                    'T_Debt', 'MV_CE', 'LEV_TD', 'Mkt_LEV', 
#                    'Market_to_Book', 'Op_profit', 'Cash_liq',
#                    'log_asset', 'log_MV_CE', 'log_sale', 
#                    'intan_cap_AT', 'intan_cap_lessgdwl_AT', 'intan_AT', 'intan_lessgdwl_AT', 
#                    'know_cap_AT', 'organ_cap_AT', 'PPENT_AT'] + industry_dummies # 'LEV_LTD', 'Mkt_LEV_CE', 'Altman_Z',

    reg_df[winsor1] = reg_df[winsor1].apply(lambda x: winsorize(x, limits=[pct1, pct1]), axis=0) # -5% -5% = 90%
    reg_df[winsor2] = reg_df[winsor2].apply(lambda x: winsorize(x, limits=[pct2, pct2]), axis=0) # -10% - 10% = 80%
#     reg_df = reg_df[winsor_cols]
    return reg_df


def get_winsorskew_df(reg_df, winsor1, winsor2, side1='right', side2='right', pct1=0.05, pct2=0.1):
    reg_df = reg_df.copy()
    if side1 == 'right':
        reg_df[winsor1] = reg_df[winsor1].apply(lambda x: winsorize(x, limits=[0, pct1]), axis=0)
    elif side1 == 'left':
        reg_df[winsor1] = reg_df[winsor1].apply(lambda x: winsorize(x, limits=[pct1, 0]), axis=0)
    
    if side2 == 'right':
        reg_df[winsor2] = reg_df[winsor2].apply(lambda x: winsorize(x, limits=[0, pct2]), axis=0)
    elif side2 == 'left':
        reg_df[winsor2] = reg_df[winsor2].apply(lambda x: winsorize(x, limits=[pct2, 0]), axis=0)
    return reg_df



# Tangibility variables - meaningful after winsorizing
# Med_Tangibility: Median tangibility per industry
# High_Tangibility: Dummy variable based on Med_Tangibility of each industry
def Tangibility_dummy(reg_df, visualize = 'True'):
    reg_df['Med_Tan'] = None
    reg_df['High_Tan'] = None
    
    FF_industry = list(reg_df['sic'].unique())
    for industry in FF_industry:
        # Calculate median tangibilities per industry
        idry_med = reg_df[reg_df['sic'] == industry]['PPENT_AT'].median()
        reg_df.loc[reg_df['sic'] == industry,'Med_Tan'] = idry_med
    
    # Handle data type, dummy vars = int, ratios = float
    reg_df['Med_Tan'] = reg_df['Med_Tan'].apply(lambda x: float(x))
    
    # High tangibility dummy variable
    reg_df['High_Tan'] = reg_df['PPENT_AT'] > reg_df['Med_Tan']
    reg_df['High_Tan'] = reg_df['High_Tan'].apply(lambda x: 1 if x == True else 0)
    
    if visualize == 'True':
        plt.figure(figsize=(15,3))
        vis_industry = reg_df['sic'].value_counts() / len(reg_df['sic']) * 100
        vis_industry.plot(kind='bar');
        plt.show()

        plt.figure(figsize=(15,3))
        vis_HiTan = reg_df['High_Tan'].value_counts() / len(reg_df['High_Tan']) * 100
        vis_HiTan.plot(kind='bar').set_xticklabels(['Low_Tan', 'Hi_Tan'], rotation=0);
        plt.show()

        plt.figure(figsize=(15,3))
        vis_indry_Hitan = reg_df.groupby('sic')['High_Tan'].value_counts() / reg_df.groupby('sic')['High_Tan'].count()
        vis_indry_Hitan.plot(kind='bar');
        plt.show()
    
    return reg_df

def Time_period_dummy(reg_df, T):
    reg_df[f'After{T}'] = None
    reg_df[f'After{T}'] = np.where(reg_df['fyear'] > T, 1, 0)
    return reg_df

def heterogeneity_vars(reg_df):
    reg_df['HiTan_INT'] = reg_df['High_Tan'] * reg_df['intan_cap_AT']
    reg_df['T2_INT'] = reg_df['After2015'] * reg_df['intan_cap_AT']
    reg_df['HiTan_T2_INT'] = reg_df['High_Tan'] * reg_df['After2015'] * reg_df['intan_cap_AT']
    return reg_df

## meaningful after winsorizing
def LEV_dummy(reg_df):
    reg_df['Med_LEV'] = None
    reg_df['Med_Mkt_LEV'] = None
    
    FF_industry = list(reg_df['sic'].unique())
    for industry in FF_industry:
        # Calculate median LEV per industry
        idry_med = reg_df[reg_df['sic'] == industry]['LEV_TD'].median()
        reg_df.loc[reg_df['sic'] == industry,'Med_LEV'] = idry_med
        
        # Calculate median Mkt_LEV per industry
        idry_mkt_med = reg_df[reg_df['sic'] == industry]['Mkt_LEV'].median()
        reg_df.loc[reg_df['sic'] == industry,'Med_Mkt_LEV'] = idry_mkt_med

    # handle data type
    reg_df['Med_LEV'] = reg_df['Med_LEV'].apply(lambda x: float(x))
    reg_df['Med_Mkt_LEV'] = reg_df['Med_Mkt_LEV'].apply(lambda x: float(x))
    
    return reg_df


def Industry_regvar_vis(reg_df, w = 12,h = 15):
    fig, ax = plt.subplots(3,2, figsize=(w,h))
    # Visualizations of Groupby industry descriptive statistics for book leverage
    plt.subplot(3, 2, 1)
    reg_df.groupby('sic')['LEV_TD'].mean()\
                                     .rename_axis(None)\
                                     .astype('float')\
                                     .plot(kind='bar')
    ax[0,0].set_title('Mean Book Leverage by Industry')
    ax[0,0].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()

    # Visualizations of Groupby industry descriptive statistics for market leverage
    plt.subplot(3, 2, 2)
    reg_df.groupby('sic')['Mkt_LEV'].mean()\
                                      .rename_axis(None)\
                                      .astype('float')\
                                      .plot(kind='bar')
    ax[0,1].set_title('Mean Market Leverage by Industry')
    ax[0,1].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()

    # Visualizations of Groupby industry descriptive statistics for intangibility
    plt.subplot(3, 2, 3)
    reg_df.groupby('sic')['intan_cap_AT'].mean()\
                                           .rename_axis(None)\
                                           .astype('float')\
                                           .plot(kind='bar')
    ax[1,0].set_title('Mean Intangible Assets / Total Assets by Industry')
    ax[1,0].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()

    # Visualizations of Groupby industry descriptive statistics for tangibility
    plt.subplot(3, 2, 4)
    reg_df.groupby('sic')['PPENT_AT'].mean()\
                                       .rename_axis(None)\
                                       .astype('float')\
                                       .plot(kind='bar')
    ax[1,1].set_title('Mean Tangible Assets / Total Assets by Industry')
    ax[1,1].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()

    # Visualizations of Groupby industry descriptive statistics for knowledge capital
    plt.subplot(3, 2, 5)
    reg_df.groupby('sic')['know_cap_AT'].mean()\
                                          .rename_axis(None)\
                                          .astype('float')\
                                          .plot(kind='bar')
    ax[2,0].set_title('Mean Knowledge Capital / Total Assets by Industry')
    ax[2,0].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()

    # Visualizations of Groupby industry descriptive statistics for organization capital
    plt.subplot(3, 2, 6)
    reg_df.groupby('sic')['organ_cap_AT'].mean()\
                                          .rename_axis(None)\
                                          .astype('float')\
                                          .plot(kind='bar')
    ax[2,1].set_title('Mean Organization Capital / Total Assets by Industry')
    ax[2,1].tick_params(axis="x",rotation=90,labelsize=10)
    plt.tight_layout()
    
    #plt.savefig('Industry_regvar_vis.png')
    return None

def outlier_winsor(reg_df, var):
    
    plt.figure(figsize=(5,3))
    plt.title('Before winsorization')
    
    reg_df[var].hist(bins=30, range=[reg_df[var].min(), reg_df[var].max()]);
    plt.figure(figsize=(5,3))
    plt.boxplot(reg_df[var]);
    non_outliers = reg_df[var][find_outliers_IQR(reg_df)[var].isnull()]
    outliers = reg_df[var][find_outliers_IQR(reg_df)[var].notnull()]
    print(len(non_outliers), len(outliers), len(non_outliers) + len(outliers), len(reg_df[var]))
    
    plt.figure(figsize=(5,3))
    plt.scatter(outliers.index, outliers);
    plt.scatter(non_outliers.index, non_outliers);
    print('skew, norm=0',skew(reg_df[var]))
    print('kurt, norm=3',kurtosis(reg_df[var]))

    if (outliers.min() < non_outliers.min()) & (non_outliers.max() < outliers.max()):
        print('WINSORIZE TOP AND BOTTOM')
        upper_percentile = stats.percentileofscore(reg_df[var], outliers[outliers > non_outliers.max()].min())
        lower_percentile = stats.percentileofscore(reg_df[var], outliers[outliers < non_outliers.min()].max())
        print('upper percentile of minimum outlier:', upper_percentile)
        print('lower percentile of minimum outlier:', lower_percentile)
        winsor_upper = np.ceil(100 - upper_percentile) / 100
        winsor_lower = np.ceil(lower_percentile) / 100
        print('winsorization upper percentile of minimum outlier:', winsor_upper)
        print('winsorization lower percentile of minimum outlier:', winsor_lower)
        winsorized = pd.DataFrame(winsorize(reg_df[var], limits=[winsor_lower, winsor_upper]))

        plt.figure(figsize=(5,3))
        winsorized.hist(bins=30);
        plt.title('After winsorization')
        plt.figure(figsize=(5,3))
        plt.boxplot(winsorized);
        plt.figure(figsize=(5,3))
        plt.scatter(outliers.index, outliers, color='red');
        plt.scatter(winsorized.index, winsorized);

        print('skew, norm=0',skew(winsorized))
        print('kurt, norm=3',kurtosis(winsorized))

    elif non_outliers.max() < outliers.min():
        print('WINSORIZE TOP')
        print(non_outliers.max(), outliers.min(), 'outliers only above:', non_outliers.max() < outliers.min())
        upper_percentile = stats.percentileofscore(reg_df[var], outliers.min())
        print('upper percentile of minimum outlier:', upper_percentile)
        winsor_upper = np.ceil(100 - upper_percentile) / 100
        print('winsorization upper percentile of minimum outlier:', winsor_upper)
        winsorized = pd.DataFrame(winsorize(reg_df[var], limits=[0, winsor_upper]))

        plt.figure(figsize=(5,3))
        winsorized.hist(bins=30);
        plt.title('After winsorization')
        plt.figure(figsize=(5,3))
        plt.boxplot(winsorized);
        plt.figure(figsize=(5,3))
        plt.scatter(outliers.index, outliers, color='red');
        plt.scatter(winsorized.index, winsorized);

        print('skew, norm=0',skew(winsorized))
        print('kurt, norm=3',kurtosis(winsorized))

    elif outliers.max() < non_outliers.min():
        print('WINSORIZE BOTTOM')
        print(outliers.max(), non_outliers.min(), 'outliers only below:', outliers.max() < non_outliers.min())
        lower_percentile = stats.percentileofscore(reg_df[var], outliers.max())
        print('lower percentile of minimum outlier:', lower_percentile)
        winsor_lower = np.ceil(lower_percentile) / 100
        print('winsorization lower percentile of minimum outlier:', winsor_lower)
        winsorized = pd.DataFrame(winsorize(reg_df[var], limits=[winsor_lower, 0]))

        plt.figure(figsize=(5,3))
        winsorized.hist(bins=30);
        plt.title('After winsorization')
        plt.figure(figsize=(5,3))
        plt.boxplot(winsorized);
        plt.figure(figsize=(5,3))
        plt.scatter(outliers.index, outliers, color='red');
        plt.scatter(winsorized.index, winsorized);

        print('skew, norm=0',skew(winsorized))
        print('kurt, norm=3',kurtosis(winsorized))

    print('check any outliers:', find_outliers_IQR(winsorized).notnull().sum()[0])