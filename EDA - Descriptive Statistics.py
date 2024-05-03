#!/usr/bin/env python
# coding: utf-8

# ## Desriptive staistics
# 
# 

# In[ ]:


#it provides a quantitative summary of a variable and the data points that comprise it.

# two categories of statics

#1. Descriptive statistics that describe the values of observations in a variable.
#2. Descriptive staistics that describe the variable spread.


# In[ ]:


#Descriptive Statistics of a Variable (observations in a variable)

#sum
#median
#mean
#max


#Descriptive Statistics of a Variable spreed

#Standard deviation
#Variance
#Counts
#Quartiles


# In[ ]:


#Descriptive statistics is used for

#1. Detecting out liers
#2. Planning data preparations requirements for machine learning
#3. Selecting features for use in machine learning


# In[43]:


import numpy as np
import pandas as pd

from pandas import Series, DataFrame




# In[44]:


cars = pd.read_csv('mtcars.csv')
cars.head()


# In[45]:


#summary statistics that describe a variable's numeric values


cars.sum()


# In[46]:


#the column model was drop since it is not a categorical column, to be able to acrry out some analysis.
cars.drop(columns=['model'], inplace=True)


# In[48]:


cars.head()


# In[51]:


cars.sum()


# In[53]:


cars.sum(axis = 1)


# In[54]:


cars.median()


# In[55]:


cars.mean()


# In[56]:


cars.min()


# In[57]:


cars.max()


# In[59]:


# to know the statistics of a particular row.
# row were the maximum value came from
mpg = cars.mpg
mpg.idxmax()


# #### Summary statistics that describe variable destination

# In[60]:


#standard deviation

cars.std()


# In[61]:


# for variance

cars.var()


# In[62]:


#to know the uniqueness in a dataset

gear = cars.gear
gear.value_counts()

#what this means is that cars 15 cars with 3 gears, 12 cars with 4 gears and 5 cars with 5 gears


# In[64]:


#for carb
cyl = cars.cyl
cyl.value_counts()


# In[66]:


cars.describe()


# In[ ]:





# #### Continuatuion of DESCRIPTIVE STATISTICS with DataDaft

# In[ ]:


#Descriptive statistics are measures that summarize important features of data, often nwitha single number.
#Is the common first step to take after cleaning and preparing data.

#Common measures are mean, mode and median.


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


#The median gives us a value that splits the data into two halves.
#the mean is a numerice average, extreme values can have a significant impact on the mean.
#In a symmetric distribution, the mean and median will be the same.
#median is also known as the 50% percentile


# In[96]:


data = pd.read_csv('mtcars.csv')
data
data.head()


# In[97]:


carz = data.drop(columns=['model'])
carz.head()
#cars.drop(columns=['model'], inplace=True)


# In[98]:


data = carz
data.head()


# In[65]:


norm_data = pd.DataFrame(np.random.normal(size=100000))
norm_data.plot(kind = 'density', figsize =(10, 10));

plt.vlines(norm_data.mean(),
          ymin = 0,
          ymax = 0.4,
          linewidth = 5.0,);

plt.vlines(norm_data.median(),
          ymin = 0,
          ymax = 0.4,
          linewidth = 2.0,
          color = 'red');


# In[ ]:


In the plot above the mean and median are both so close to zero that the red median line lies on top of the thicker blue line drawn at the mean.
In skewed distributions, the meand tends to get pulled in the direction of the skew, while the median tends to resist the effects of the skew.


# In[81]:


skewed_data = pd.DataFrame(np.random.exponential(size=100000))
                          
skewed_data.plot(kind = 'density',
                 figsize = (10, 10),
                 xlim = (-1, 5));
                        
plt.vlines(skewed_data.mean(),
           ymin = 0,
           ymax = 0.8,
           linewidth = 5.0);
                        
plt.vlines(skewed_data.median(),
           ymin = 0,
           ymax = 0.8,
           linewidth = 2.0,
           color = 'red');


# In[ ]:


The mean is influenced heavily by outliers, while thenmedian resists the influence of outliers.


# In[85]:


norm_data = np.random.normal(size=50)
outliers = np.random.normal(15, size = 3)
combined_data = pd.DataFrame(np.concatenate((norm_data, outliers), axis = 0))

combined_data.plot(kind = 'density',
                  figsize = (10, 10),
                  xlim = (-5, 20));

plt.vlines(combined_data.mean(),
          ymin = 0,
          ymax = 0.2,
          linewidth = 5.0);

plt.vlines(combined_data.median(),
          ymin = 0,
          ymax = 0.2,
          linewidth = 2.0,
          color = 'red');


# In[ ]:


Since the median tends to resist the effects of skewness and outliers, it is known as a 'robust statistic'.
The median generally gives a better sense of the typical value in a distribution with significant skew or outliers.
The mode of a variable is simply the value that appears most frequently


# In[100]:


data.mode()


# ### Measures of Spread

# In[ ]:


Measures of spread(dispersion) are statistics that describe how data varies. While measures of center gives us an idea of the typical value,measures of spread gives us a sense
of how much the data tends to diverge from typical value.
Example of range is Range which is the distance between the max and min obervations.


# In[101]:


#Range
max(data['mpg']) - min(data['mpg'])


# In[102]:


#The median of a data set is the 50%tile of the dataset

five_num = [data ['mpg'].quantile(0),
            data ['mpg'].quantile(0.25),
            data ['mpg'].quantile(0.50),
            data ['mpg'].quantile(0.75),
            data ['mpg'].quantile(1)]

five_num

#They are known as the five number summary.


# In[104]:


data['mpg'].describe()


# In[107]:


#Interquartile(IQR) range is  d distance between the 3rd and 1st quartile.

data['mpg'].quantile(0.75) - data['mpg'].quantile(0.25)


# In[109]:


#variance  average of the squared deviations(differences) from the mean

data['mpg'].var()


# In[111]:


#Standard deviation is the square root of the variance
data['mpg'].std()


# In[ ]:


Skewness is a statistics that measure the skew or symmetric of a distribution
Kurtos is how much data is in the tails of the distribution versus the center.

Skewness cubing deviation while  Kurtosis raising those deviation to the forth power


# In[112]:


data['mpg'].skew()   #skew


# In[113]:


data['mpg'].kurt()  #kurt


# In[ ]:


for skew and kurt don't really mean alot if we don't have frame of reference for how much skew and kurt it is.


# In[ ]:





# In[ ]:




