# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:19:54 2022

@author: ordersofmagnitude
"""
import pandas as pd
import random
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy


from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.validation import array_like
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Compute Seasonal Index
from dateutil.parser import parse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from xgboost import XGBRegressor




#time series analytics

class preprocessing:
    
    def drop_redundant(self, dfs):
        for df in dfs:
            for col in df.columns:
                if "Unnamed" in col:
                    df.drop(col, inplace = True, axis = 1)
                    
    
    def null_filler(self, dfs):
    
        '''takes a dictionary of dataframes, fills null values automatically using KNNImputer'''
        if type(dfs) == dict:
            for name, df in dfs.items():
                if df.isnull().values.any():
                    imputer = KNNImputer(n_neighbors = 5)
                    df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns, index = df.index)
                    dfs[name] = df
                else:
                    pass
                
        elif isinstance(dfs, pd.DataFrame):
            
            if dfs.isnull().values.any():
                imputer = KNNImputer(n_neighbors = 5)
                dfs = pd.DataFrame(imputer.fit_transform(dfs),columns = dfs.columns, index = dfs.index)
                
        return dfs
    
    #downsampling

    def downsampler(self, dfs, freq):
        
        '''takes a dictionary of dataframes, automatically downsamples them and fills null values using KNNImputer'''
        
        df_monthlies = {}
        
        if type(dfs) == dict:
        
            for name, df in dfs.items():

                df.columns = df.columns.str.lower()
                df_copy = copy.deepcopy(df)
                df_copy["date"] = pd.to_datetime(df_copy["date"])
                df_copy.set_index("date", inplace = True)
                resample = df_copy.resample(freq)
                df_monthly = resample.mean()
                df_monthlies[name] = df_monthly

            for name, df in df_monthlies.items():
                if df.isnull().values.any():
                    imputer = KNNImputer(n_neighbors = 5)
                    df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns, index = df.index)
                    df_monthlies[name] = df
                    
            return df_monthlies
                    
        elif isinstance(dfs, pd.DataFrame):
            
            dfs.columns = dfs.columns.str.lower()
            df_copy = copy.deepcopy(dfs)
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            df_copy.set_index("date", inplace = True)
            resample = df_copy.resample(freq)
            df_monthly = resample.mean()

            if df_monthly.isnull().values.any():
                imputer = KNNImputer(n_neighbors = 5)
                df = pd.DataFrame(imputer.fit_transform(df_monthly),columns = df_monthly.columns, index = df_monthly.index)
                    
            return df

    
    #drop data by date

    def date_aligner(self, dfs, date):
        
        for name, df in dfs.items():
            
            #exclude all indexes before a certain date
            df_aligned = df[df.index >= date]
            dfs[name] = df_aligned
            
        return dfs
    
    #rename columns before merging
    def column_renamer(self, dfs):
        
        dfs_copy = {}
        
        for name, df in dfs.items():
        
            df_copy = df.copy()
    
            for i, col in enumerate(df_copy.columns):
                col = f"{name}_" + col
                df_copy.columns.values[i] = col
            
            dfs_copy[name] = df_copy            
            
        return dfs_copy
    

    def one_price_per_date(self, df, mode = "high"):
        
        #get a OHLC dataframe
       
        price_df = pd.DataFrame()
        average_prices = []
            
        for date in df["date"].unique():
            tmp = df[df["date"] == date]
            
            if mode == "high":
                
                #if more than 1 row, randomly select one
                
                price = tmp["price_sgd"].max()
                
            elif mode == "low":
                price = tmp["price_sgd"].min()
                
            
            average_price = tmp["price_sgd"].mean()
            average_prices.append(average_price)
            
            get_price = tmp[tmp["price_sgd"] == price]
            
            if len(get_price) > 1:
                get_price = get_price.iloc[random.randint(0, len(get_price)-1)]
                get_price = pd.DataFrame(get_price)
                get_price = pd.DataFrame([get_price[get_price.columns[0]]], columns = get_price.index)
                
            price_df = pd.concat([price_df, get_price], axis = 0)
        
        price_df["average"] = average_prices
            
        return price_df


    def OHLC_parser(self, dfs):
        
        #get a OHLC dataframe
        
        parsed = {}
       
        for name, df in dfs.items():
            no_of_auctions = []
                
            for date in df["date"].unique():
                tmp = df[df["date"] == date]
                no_of_auctions.append(len(tmp))
                
            df_high = self.one_price_per_date(df, mode = "high")
            df_high.rename({"price_sgd": "high"}, axis = 1, inplace = True)
            
            df_low = self.one_price_per_date(df, mode = "low")
            df_low.rename({"price_sgd": "low"}, axis = 1, inplace = True)
            
            #reset index
            
            for df in [df_high, df_low]:
                df.reset_index(inplace = True)
            
            ohlc_df = df_high.copy()
            ohlc_df["low"] = df_low["low"]
            ohlc_df["average"] = df_high["average"]
            
            for col in ohlc_df.columns:
                if col in ["price_gbp", "price_per_volume"]:
                    ohlc_df.drop(col, axis = 1, inplace = True)
                
            parsed[name] = ohlc_df

        return parsed
    
    
    #generate lagged data
    
    def generate_lagged_data(self, dfs, col, n, freq):
    
        lagged = {}
        
        if type(dfs) == dict:
        
            for name, df in dfs.items():
                df_copy = copy.deepcopy(df)
    
                for i in range(1, n+1):
                    #positive: generates lagged data (shifts previous month data forwards)
                    #ensure that datetimeindex is in ascending order before implementing
    
                    if not isinstance(df_copy.index, pd.DatetimeIndex):
                        dfs_copy = self.downsampler(dfs, freq)
                        return self.generate_lagged_data(dfs_copy, col, n, freq)
    
                    df_copy[f"lagged {i}"] = df_copy[col].shift(i)  
    
                lagged[name] = df_copy
        
        elif isinstance(dfs, pd.DataFrame):
            
            df_copy = copy.deepcopy(dfs)
    
            for i in range(1, n+1):
    
                if not isinstance(df_copy.index, pd.DatetimeIndex):
                    dfs_copy = self.downsampler(dfs, freq)
                    return self.generate_lagged_data(dfs_copy, col, n, freq)
    
                df_copy[f"lagged {i}"] = df_copy[col].shift(i)  
    
            lagged = df_copy
            
        lagged = self.null_filler(lagged)
        return lagged
    
    #generate time dummies


class hwes:
    
    def decompose(self, dfs, col, univariate):
    
    #plot grid for univariate
        if univariate == True:
    
            for name, df in dfs.items():
                decompose_result = seasonal_decompose(df[col], model= "multiplicative")
                
                plt.figure(figsize = (8,5))
                decompose_result.trend.plot()
                plt.xticks(rotation = 90)
                plt.title(f"{name} trend")
                
                plt.figure(figsize = (8,5))
                decompose_result.seasonal.plot()
                plt.title(f"{name} seasonality")
                plt.xticks(rotation = 90)
                
                plt.figure(figsize = (8,5))
                decompose_result.resid.plot()
                plt.title(f"{name} residuals")
                plt.xticks(rotation = 90)
                plt.show()
    
    
    def decompose_trend(self, dfs, col, univariate, extrapolate):
    
    #same function, except that you can extrapolate trends
        if univariate == True:
    
            for name, df in dfs.items():
                decompose_result = seasonal_decompose(df[col], model= "multiplicative",
                                                      extrapolate_trend = extrapolate)
                
                plt.figure(figsize = (8,5))
                decompose_result.trend.plot()
                plt.xticks(rotation = 90)
                plt.title(f"{name} {col} trend")
                
                plt.figure(figsize = (8,5))
                decompose_result.seasonal.plot()
                plt.title(f"{name} {col} seasonality")
                plt.xticks(rotation = 90)
                
                plt.figure(figsize = (8,5))
                decompose_result.resid.plot()
                plt.title(f"{name} {col} residuals")
                plt.xticks(rotation = 90)
                plt.show()
                
                
            
    def hwes_plot(self, dfs, col):
    
    #convert this to grid plotter
        
        for name, df in dfs.items():
            df2 = df.copy()
            df2["HWES2_MUL"] = ExponentialSmoothing(df[col],trend="mul").fit().fittedvalues
            df2[[col, "HWES2_MUL"]].plot(title=f"Holt Winters Double Exponential Smoothing for {name} {col}: Multiplicative Trend");   



class time_series:
    
    
    #models
    
    def ARIMA_model(self, df, col, pdq = (1,1,2)):
        
        '''fits an ARIMA model, returns model'''
        
        model = ARIMA(df[col], order=pdq).fit()
        return model
    
    
    def VAR_model(df, d, n, best_p = False):
    
    
        df_diff = df.diff(d)
        df_diff.dropna(inplace = True)
        
        model = VAR(df_diff)
        
        results = {"VAR P": [], "AIC": [], "BIC": [], "FPE": [], "HQIC": []}
        
        
        if best_p == False:
            
            #directly fit the parameters
            
            result = model.fit(n)
            results["VAR P"].append(n)
            results["AIC"].append(result.aic)
            results["BIC"].append(result.bic)
            results["FPE"].append(result.fpe)
            results["HQIC"].append(result.hqic)
            
        else:
            
            #search for best p
            
            for i in range(1, n):
                result = model.fit(n)
                results["VAR P"].append(n)
                results["AIC"].append(result.aic)
                results["BIC"].append(result.bic)
                results["FPE"].append(result.fpe)
                results["HQIC"].append(result.hqic)
        
        
        return pd.DataFrame(results), result
    
    
    #diagnostics
    
    def adf_tester(self, dfs, col, multivariate):
        
        '''testing for stationarity'''
    
        #two ways to input:
        #multivariate dataset: col = list of cols, df = merged df
        #univariate dataset: dfs = list of dfs, col = the same col
        
        
        if multivariate == True:
            
            adf = {"Feature": [],
                   "ADF Test Statistic": [],
                  "P-Value": []}
            
            for c in col:
                c_diff = dfs[c].diff(1)
                c_diff.dropna(inplace = True)
                
                results = adfuller(c_diff.values)
                adf["Feature"].append(c)
                adf["ADF Test Statistic"].append(results[0])
                adf["P-Value"].append(results[1])
            
        
        else:
            
            adf = {"Model": [],
                   "ADF Test Statistic": [],
                  "P-Value": []}
            
            for name, df in dfs.items():
                
                df_diff = df[col].diff(1)
                df_diff.dropna(inplace = True)
                
                results = adfuller(df_diff.values)
                adf["Model"].append(name)
                adf["ADF Test Statistic"].append(results[0])
                adf["P-Value"].append(results[1])
        
        return pd.DataFrame(adf)
    
    
    def dw_tester(self, models, multivariate):
    
        '''takes a dict of fitted time series as input,
        returns dw test statistic/p-value'''
    
    #two ways to input:
    #multivariate dataset: col = list of cols, df = merged df
    #univariate dataset: dfs = list of dfs, col = the same col/
    
        if multivariate == True:
            
            dw_df = {"Feature": [],
                     "P-Values": []}
            
            residuals = models.resid
            
            for col in residuals.columns:
                dw_df["Feature"].append(col)                             
                dw_df["P-Values"].append(durbin_watson(residuals[col]))
        
        
        else:
    
            dw_df = {"Model": [],
                     "P-Values": []}
            
            for name, model in models.items():
                dw_df["Model"].append(name)
                dw_df["P-Values"].append(durbin_watson(model.resid))
            
        return pd.DataFrame(dw_df)
    

    def homoscedascity(self, model, multivariate, fit = False, plot = True):
        
        '''plots residuals to check if they are homoscedascitic'''
        
        if multivariate == True:
            residuals = pd.DataFrame(model.resid)
            
            #loop through each column and plot the subplot
        
        
        else:        
            if fit == False:
                residuals = pd.DataFrame(model.resid)
            
            if plot == True:
                fig, ax = plt.subplots(1,2)
                residuals.plot(title = "Residuals", ax = ax[0])
                residuals.plot(kind = "kde", title = "Density", ax = ax[1])
                plt.show()
                
            else:
                
                #debug this
                
                #white's heteroscedascity test requires exog to have at least
                #2 columns where 1 is a constant
                
                white_test = self.het_white(residuals, model.model.exog)
                
                labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
                  
                return pd.DataFrame(dict(zip(labels, white_test)))
            
    def het_white(self, resid, exog):
        """
        White's Lagrange Multiplier Test for Heteroscedasticity.
    
        Parameters
        ----------
        resid : array_like
            The residuals. The squared residuals are used as the endogenous
            variable.
        exog : array_like
            The explanatory variables for the variance. Squares and interaction
            terms are automatically included in the auxiliary regression.
    
        Returns
        -------
        lm : float
            The lagrange multiplier statistic.
        lm_pvalue :float
            The p-value of lagrange multiplier test.
        fvalue : float
            The f-statistic of the hypothesis that the error variance does not
            depend on x. This is an alternative test variant not the original
            LM test.
        f_pvalue : float
            The p-value for the f-statistic.
    
        Notes
        -----
        Assumes x contains constant (for counting dof).
    
        question: does f-statistic make sense? constant ?
    
        References
        ----------
        Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
        Greene 5th, example 11.3.
        """
        x = array_like(exog, "exog", ndim=2)
        y = array_like(resid, "resid", ndim=2, shape=(x.shape[0], 1))
        if x.shape[1] < 2:
            raise ValueError("White's heteroskedasticity test requires exog to"
                             "have at least two columns where one is a constant.")
        nobs, nvars0 = x.shape
        i0, i1 = np.triu_indices(nvars0)
        exog = x[:, i0] * x[:, i1]
        nobs, nvars = exog.shape
        assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
        resols = OLS(y ** 2, exog).fit()
        fval = resols.fvalue
        fpval = resols.f_pvalue
        lm = nobs * resols.rsquared
        # Note: degrees of freedom for LM test is nvars minus constant
        # degrees of freedom take possible reduced rank in exog into account
        # df_model checks the rank to determine df
        # extra calculation that can be removed:
        assert resols.df_model == np.linalg.matrix_rank(exog) - 1
        lmpval = stats.chi2.sf(lm, resols.df_model)
        return lm, lmpval, fval, fpval


    
    #multivariate diagnostics
    
    def grangers(self, data, variables, maxlag, test='ssr_chi2test', verbose=False):    
    
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table 
        are the P-Values. P-Values lesser than the significance level (0.05), implies 
        the Null Hypothesis that the coefficients of the corresponding past values is 
        zero, that is, the X does not cause Y can be rejected.
    
        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose:
                    print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        
        return df

    
    
    
class Valuation:
    
    #the xgbr feels messy, especially the valuation part
    
    def xgbr_modeller(self, model_name, x, y_col, to_drop = [], to_value = [], grid_search = True, model = None,
                      log = True, robustscaler = True):
        
        #fit the model
        
        
        x_copy = copy.deepcopy(x)
        y = x_copy[y_col]
        
    
        for col in (to_drop + [y_col]):
            if col in x_copy.columns:
                x_copy.drop(col, inplace = True, axis = 1)
        
        if log == True:
    
            x_train, x_test, y_train, y_test = train_test_split(x_copy, np.log(y), random_state = 42)
        else:
            
            x_train, x_test, y_train, y_test = train_test_split(x_copy, y, random_state = 42)
        
        if robustscaler == True:
        
            rs = RobustScaler()
            x_train = rs.fit_transform(x_train)
            x_test = rs.transform(x_test)
        
        
        if grid_search == True:
        
            default_parameters = {

                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 8, 10, None],
                    #"min_child_weight": [1, 5, 10, None],
                    "gamma": np.linspace(0, 1, 6),
                    "reg_alpha": np.linspace(1e-6, 1e-3, 4),
                    "random_state": [42],
                    "grow_policy": ["depthwise", "lossguide"]
                        }


            xgbr = GridSearchCV(XGBRegressor(), default_parameters, cv=5, n_jobs=-1, scoring= "neg_root_mean_squared_error")
            xgbr.fit(x_train, y_train)
            xgbr_best = xgbr.best_estimator_
            xgbr_best.save_model(f"models/{model_name}.json")
    
    
        else:
            xgbr_best = model.fit(x_train, y_train)
    
        #if there isn't any data to value, return the fitted model
        if len(to_value) == 0:
            
            #save the model into a pickle format
            
            #best params
            #autoscorer
            
            scores = pd.DataFrame()
            
            train_r2 = xgbr_best.score(x_train, y_train)
            test_r2 = xgbr_best.score(x_test, y_test)
      
            train_rmse = self.rmse(y_train, xgbr_best.predict(x_train))
            test_rmse = self.rmse(y_test, xgbr_best.predict(x_test))
          
            metrics = [f"{model_name}", train_r2, test_r2, train_rmse, test_rmse]
              
            scores = pd.concat([scores, pd.DataFrame([metrics],
                                                       columns = ["Model Name", "train_r2", "test_r2", "train_rmse", "test_rmse"]
                                                      )], axis = 0)
        
    
            
            
            return xgbr_best, scores
    
    
        
        
    
        
    def rmse(self, y, y_pred):
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return rmse
      
        
      
    def auto_linear_scorer(self, x, y, models):
        
        #gets a dict of models with name & models, scores & returns scores
          
        x_train, x_val, y_train, y_val = train_test_split(x, np.log(y), random_state = 42)
        rs = RobustScaler()
        x_train_rs = rs.fit_transform(x_train)
        x_val_rs = rs.transform(x_val)
          
        scores = pd.DataFrame()
          
        for name, model in models.items():
              
            try:
                train_r2 = model.best_estimator_.score(x_train_rs, y_train)
                val_r2 = model.best_estimator_.score(x_val_rs, y_val)
          
                train_rmse = self.rmse(y_train, model.best_estimator_.predict(x_train_rs))
                val_rmse = self.rmse(y_val, model.best_estimator_.predict(x_val_rs))
              
            except:
                train_r2 = model.score(x_train_rs, y_train)
                val_r2 = model.score(x_val_rs, y_val)
          
                train_rmse = self.rmse(y_train, model.predict(x_train_rs))
                val_rmse = self.rmse(y_val, model.predict(x_val_rs))
              
            metrics = [f"{name}", train_r2, val_r2, train_rmse, val_rmse]
              
            scores = pd.concat([scores, pd.DataFrame([metrics],
                                                       columns = ["Model Name", "train_r2", "val_r2", "train_rmse", "val_rmse"]
                                                      )], axis = 0)
            
        return scores
    
    
    def get_impt_features(self, df, y, to_drop, models):
        
        #multiple models for the same dataset
        
        df_copy = copy.deepcopy(df)
        
        for col in df_copy.columns:
            
            #add shit to array
            
            if col in set(to_drop + [y]):
                df_copy.drop(col, axis = 1, inplace = True)

        results = pd.DataFrame()
        

        for name, model in models.items():

            try:
                importances = np.exp(model.coef_)
                df_features = (
                            pd.DataFrame(
                                zip(df_copy.columns, importances),
                                columns=[name, f"{name} Importances"],
                          )
                            .sort_values(f"{name} Importances", ascending=False)
                            .reset_index(drop=True)
                        )
                results = pd.concat([results, df_features], axis=1)

            except:
                pass

            try:
                importances = model.feature_importances_
                df_features = (
                            pd.DataFrame(
                                zip(df_copy.columns, importances),
                                columns=[name, f"{name} Importances"],
                          )
                            .sort_values(f"{name} Importances", ascending=False)
                            .reset_index(drop=True)
                        )
                results = pd.concat([results, df_features], axis=1)

            except:
                pass

        return results
    
    
        
    def plot_impt_features(self, df, x_name, y_name, n = 20):
                
        sns.catplot(x = x_name,
                              y = y_name,
                              data = df.head(n),
                              kind = 'bar',
                              height = 8,
                              aspect = 2,
                              palette = "icefire"
                         )

        plt.title('Coefficients', fontsize = 24)
        plt.xlabel("Coefficients", size = 20)
        plt.ylabel("Features", size = 20)
        plt.show()
        
        
