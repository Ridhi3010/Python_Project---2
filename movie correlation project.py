#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import seaborn as sns 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[3]:


movies = pd.read_csv("C:\\Users\\HP\\Downloads\\archive (1)\\movies.csv")


# In[4]:


movies


# In[5]:


movies.head()


# In[6]:


for col in movies.columns:
    pct_missing = np.mean(movies[col].isnull())
    print('{} - {}%'.format(col,pct_missing))
    


# In[7]:


#datatypes 
movies.dtypes


# In[8]:


movies['budget']= movies['budget'].astype('int64')
 
movies['gross']= movies['gross'].astype('int64')


# In[9]:


movies


# In[10]:


year_correct = movies['released'].astype(str).str


# In[11]:


year_correct


# In[12]:


movies


# In[6]:


movies = movies.sort_values(by= ['gross'], inplace = False ,ascending =False)


# In[16]:


pd.set_option('display.max_rows', None)


# In[17]:


#drop duplicate

movies['company'].drop_duplicates().sort_values(ascending = False)


# In[ ]:


# budget high correlation
#company high correlation 


# In[8]:


#scatter plot with budget with gross revenue 

plt.scatter(x=movies['budget'], y =movies['gross'])

plt.title('Budget vds Gross Earnings')

plt.xlabel('Gross earnings')

plt.ylabel('budget for flim')
plt.show()


# In[7]:


movies.head()


# In[7]:


#plot the bugdet vs gross using seaborn 

sns.regplot(x = 'budget', y ='gross', data = movies, scatter_kws = {"color" : "red"}, line_kws ={"color":"blue"})


# In[9]:


# correlation 

movies.corr(method ='pearson')#pearson , kendal , spearman


# In[11]:


movies.corr(method = 'kendall')


# In[ ]:


# high corr between budget and gross 


# In[18]:


correlation_matrix = movies.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation matrix for numeric feature')

plt.xlabel('Movies features ')

plt.ylabel('Movies features')


plt.show()


# In[19]:


# looking at company 


movies.head()


# In[22]:


movies_numerized =movies

for col_name in movies_numerized.columns :
    if(movies_numerized[col_name].dtype == 'object'):
        movies_numerized[col_name] = movies_numerized[col_name].astype('category')
        movies_numerized[col_name] = movies_numerized[col_name].cat.codes 


movies_numerized
    
    
    


# In[23]:


correlation_matrix = movies_numerized.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation matrix for numeric feature')

plt.xlabel('Movies features ')

plt.ylabel('Movies features')


plt.show()


# In[24]:


movies_numerized.corr()


# In[25]:


correlation_mat = movies_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[26]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[27]:


high_corr = sorted_pairs[(sorted_pairs) >0.5]

high_corr


# In[ ]:


#votes and budget have the highest correlation to gross earning 

#company has low correlation 
#i was wrong 

