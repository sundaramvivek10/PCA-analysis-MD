import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

analysisdata = pd.read_csv('pcadata.dat', delimiter="\s+", header=None)
analysisdata.columns = ["gr dpp", "ga dpp", "gr py", "ga py", "gr tph", "ga tph", "time"]
features = ["ga dpp", "ga py", "ga tph", ]
observation = ["time"]

# numpy arrays
x = analysisdata.loc[:, features].values #.values gives a np dataframe. No need to do .to_numpy
print(x)
y = analysisdata.loc[:, observation].values
#x = StandardScaler().fit_transform(x)
pd.DataFrame(data=x, columns=features)
time = pd.DataFrame(data=y, columns=observation)
pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
finalDf = pd.concat([principalDf, time], axis=1)
#print (finalDf.head())

fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(1,1,1)
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('g(r)-DPP', fontsize=15)
ax.set_ylabel('g(r)-Py', fontsize=15)
ax.set_zlabel('g(r)-TPH', fontsize=15)
ax.set_title('3 Component g(r)', fontsize=20)
coolingtimes = ["100ns", "500ns", "2ms"]
colors = ['r', 'g', 'b']
for coolingtime, color in zip(coolingtimes, colors):
    indicesToKeep = finalDf['time'] == coolingtime
    # ax.scatter(finalDf.loc[indicesToKeep, 'PC2'], finalDf.loc[indicesToKeep,
    #            'PC1'], finalDf.loc[indicesToKeep, 'PC3'], c=color)
    ax.scatter(analysisdata.loc[indicesToKeep, 'gr dpp'], analysisdata.loc[indicesToKeep,
               'gr py'], analysisdata.loc[indicesToKeep, 'gr tph'], c=color)
    ax.legend(coolingtimes)
    ax.autoscale()
    fig.savefig("test.png")

    # Factor Analysis
    # analysisdata.drop(observation, axis=1, inplace=True)

    chi_square_value, p_value = calculate_bartlett_sphericity(x)
    kmo_all, kmo_model = calculate_kmo(x)
    print(chi_square_value)
    print(p_value)
    print(kmo_model)
    fa = FactorAnalyzer(rotation='varimax', n_factors=2)
    fa.fit(x)
    ev, v = fa.get_eigenvalues()
    print(fa.loadings_)
    print(fa.get_communalities())
    #fa = FactorAnalysis(n_components=2)
