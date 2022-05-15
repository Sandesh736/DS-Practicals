import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#%%
df=sns.load_dataset('titanic')
print(df.head())
print(df.columns)
#sns.distplot(df["fare"],kde=False,bins=10)
#%%
#sns.jointplot(x="age",y="fare",data=df)
#%%

#sns.barplot(x="sex",y="age",data=df)
#%%

#sns.boxplot(x="sex",y="age",data=df)
#%%

#sns.violinplot(x="sex",y="age",data=df)
#%%
#sns.distplot(df["age"],kde=False,bins=10)

#%%
sns.catplot(x="embarked",hue="survived",kind="count",col="pclass",data=df)
#sns.barplot(x='fare',y='survived',hue='survived',data=df)