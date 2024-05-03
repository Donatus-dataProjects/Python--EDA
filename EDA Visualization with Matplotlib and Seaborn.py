#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib seaborn --upgrade --quite')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#magic command informs jupyter that you want your grap to show as output below the cells not as pop up


# In[3]:


using line charts; it displays information as a seres of data points or makers, connected by straight lines.
country-- Kanto 
years 6yrs 
intons per hectare



# In[59]:


yield_apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931]


# In[60]:


plt.plot(yield_apples);


# In[61]:


#Customizing it to look better


# In[62]:


years = [2010, 2011, 2012, 2013, 2014, 2015]
yield_apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931]


# In[63]:


plt.plot(years, yield_apples);


# In[64]:


#Axis Labes


# In[65]:


plt.plot(years, yield_apples)
plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)');


# ### Plotting multiple lines in the same graph

# In[66]:


years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.940, 0.945, 0.947]
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.909, 0.910, 0.913, 0.918]


# In[67]:


plt.plot(years, apples)
plt.plot(years, oranges)
plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')
plt.legend(['Apples', 'Oranges'])
plt.show()


# ### Chart Title and Legend

# In[68]:


plt.plot(years, apples)
plt.plot(years, oranges)

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yield in Kanto')
plt.legend(['Apples', 'Oranges']);


# ### Line Markers 

# In[69]:


plt.plot(years, apples, marker = 'x')
plt.plot(years, oranges, marker = 'o')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yield in Kanto')
plt.legend(['Apples', 'Oranges']);


# In[70]:


plt.plot(years, apples, marker = 'x', c = 'blue', ls = '--', lw = 2,  ms = 8)
plt.plot(years, oranges, marker = 'o', c = 'red', ms = 10, alpha = .5 )

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yield in Kanto')
plt.legend(['Apples', 'Oranges']);


# In[ ]:





# ## Changing the Figure Size

# In[71]:


plt.figure(figsize = (8, 5))

plt.plot(years, oranges, 'or')
plt.title('Yield of oranges (tons per hectare)')


# fmt arguement provides a shortcut for specifying the line style, marker and line color. It can be provided as the third agruement to plt.plot
# fmt = 'o-r'
# fmt = 'o--b'

# In[72]:


plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# In[ ]:


#If you want to have only markers and no line you have to remove the '--' in the fmt styling.


# In[73]:


plt.plot(years, apples, 'sb')
plt.plot(years, oranges, 'or')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# #### changing the figure size

# In[74]:


plt.figure(figsize=(5, 5))
plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# ## Improving Default Styles Using Seaborn

# In[75]:


sns.set_style('whitegrid')  #it gives a grid to your chat.


# In[76]:


plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# In[77]:


sns.set_style('darkgrid')     #using a darkgrid background

plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# In[78]:


plt.plot(years, apples, 'sb')
plt.plot(years, oranges, 'or')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# ### Editing default styles
# 
# You can also edit default styles directly by modifying the matplotlip rcparams dictionary. ie
# 
# matplotlib.rcParams

# In[82]:


import matplotlib


# In[83]:


matplotlib.rcParams   #matplotlib dictionary


# In[84]:


matplotlib.rcParams['font.size']= 10.0
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['figure.figsize'] = (15, 8)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

plt.plot(years, apples, 'sb')
plt.plot(years, oranges, 'or')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title('Crops Yields in Kanto')
plt.legend(['Apples', 'Oranges']);


# ### Scatter Plot
# 
# 
#  A satter plot is used to visualize the relationship between two variable as points on a two dimensional grid.
#  
# A line plot is used to represent a bunch of values in a sequence while 
# 

# In[6]:


import pandas as pd


# In[7]:


flowers = pd.read_csv('flowers.csv')
flowers


# #### Dropping a column without affecting the other columns
# 
# #flowerz = pd.read_csv('flowers.csv')
# #flowerz
# 
# #flower.drop(columns = 'species')lowers.csv')
# 
# #flower = flowerz.drop(columns='species')
# #flower.to_csv('flowers.csv', index=False)
# flower
# 

# In[8]:


flowers.describe()


# In[85]:


#to check the species of the flower


flowers.species.unique()


# In[10]:


#import pandas as pd

# Assuming 'flowers' DataFrame already contains the data
flowers.plot(x='sepal length', y='sepal width');
#sns.set_style('darkgrid')   
#plt.xlabel('Sepal Length')
#plt.ylabel('sepal width')
#plt.title('Sepal Length vs. Width Length (Color-mapped by Petal Length)')
#plt.show();


# In[ ]:





# In[11]:


sns.scatterplot(x='sepal length', y='sepal width', data = flowers);


# ### Hues
# 
# Adding Hues to the plot
# Hues is used to created the size of the point.

# In[13]:


sns.scatterplot(x='sepal length', y='sepal width', hue='species', s = 80,  data = flowers);


# ### Relationship
# #3 flowers
# when sepal length low width is high compare to other flowers.
# As flowers grows the sepal width and length grows proportionally.
# 
# #Similar trend for versicolor 
# 
# #Virginica long sepal and more sperate out

# In[14]:


#Making figure to be bigger using sns

plt.figure(figsize=(12, 6))
plt.title('Sepal Dimension')

sns.scatterplot(x='sepal length', y='sepal width', hue = 'species', s=100, data = flowers);


# In[ ]:





# ## Histogram
# 
# Is a distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin

# In[15]:


flowers['sepal length']


# In[16]:


flowers.describe()['sepal length']


# In[17]:


flowers.describe()['petal width']


# In[18]:


flowers.describe()


# In[19]:


plt.title('Distribution of Sepal Width')

plt.hist(flowers['sepal width'])
plt.xlabel('Sepal Width')
plt.ylabel('Frequency');


# In[20]:


#Specifying the number of bins

plt.hist(flowers['sepal width'], bins = 5);


# Specifying the points to which  bins should be created.
# For bins to be created you need to import numpy

# In[22]:


import numpy as np
np.arange(2, 5, 0.25)


# In[ ]:





# In[23]:


plt.hist(flowers['sepal width'], bins=np.arange(2, 5, 0.25))
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Width');


# ##### 

# In[24]:


plt.hist(flowers['sepal width'], bins=[1, 3, 4, 4.5])
#plt.xlabel('Sepal Width')
#plt.ylabel('Frequency')
plt.title('Distribution of Sepal Width');
#plt.show()

#the bin of unequal sizes starting from 1 - 3 and is of lenght 2. then 3-4 thats 
#bin lenght of 1. and the last and is 1/2 ihalf length 


# In[ ]:





# ### Drawing multiple histogram in a single chart

# In[25]:


setosa = flowers[flowers.species == 'setosa']
versicolor = flowers[flowers.species == 'versicolor']
virginica = flowers[flowers.species == 'virginica']


# In[26]:


flowers.species


# In[27]:


#drawing histogram ontop of each other. to do that you set lower opacity for each of the histogram.


# In[28]:


setosa = flowers[flowers.species == 'setosa']
versicolor = flowers[flowers.species == 'versicolor']
virginica = flowers[flowers.species == 'virginica']


# In[29]:


plt.hist(setosa['sepal width'],alpha=0.4, bins= np.arange (2, 5, 0.25))
plt.hist(versicolor['sepal width'], alpha=0.4, bins=np.arange (2, 5, 0.25));
plt.title('Flowers Distribution');
plt.legend(['setosa','versicolor' ]);


# ### Stacking histogram ontop of one another
# 
# To do this you pass in the columns you want to plot as a list.

# In[30]:


plt.title('Flowers Distribution');
plt.hist([setosa['sepal width'],versicolor['sepal width'], virginica['sepal width']],
       bins = np.arange(2, 5, 0.25),
       stacked = True);
plt.legend(['setosa', 'versicolor', 'virginica']);


# In[ ]:





# ### Bar Chart
# 
# It is similar to line line chart.

# In[31]:


get_ipython().run_line_magic('pip', 'install seaborn')


# In[32]:


import seaborn as sns


# In[33]:


years = range(2000, 2006)
apples = [0.35, 0.6, 0.9, 0.8, 0.65, 0.8]
oranges = [0.4, 0.8, 0.9, 0.7, 0.6, 0.8]


# In[34]:


#plt.plot(years, apples),
plt.plot(years, oranges);


# In[35]:


plt.bar(years,(oranges));


# In[36]:


plt.bar(years,(oranges));
plt.plot(years, oranges,  'o--r');
plt.title('Yield of Oranges');


# How to stacked bars on top of one another.
# use the bottom argument, to achieve this
# plt.bar

# In[37]:


plt.bar(years, apples);
plt.bar(years, oranges, bottom = apples);
plt.title('Yield of Fruits');
plt.legend(['apples', 'oranges']);


# Dataset that contains information about customers

# In[86]:


tips = sns.load_dataset('tips')
tips


# In[55]:


#Performing groupby with the data

avgbill = tips.groupby('day')[['total_bill']].mean()
avgbill


# In[54]:


#Performing groupby with the data

tipz = tips.groupby('day')[['tip']].mean()
tipz


# In[242]:


plt.bar(tipz.index, tipz.tip);
plt.title('Tips');


# In[245]:


plt.bar(avgbill.index, avgbill.total_bill );
plt.title('Bill')


# To draw a bar chart to visualize how the average bill amount varies across different days of the week. 
# one way to do this is to compute the day-wise averages and then use plt.bar

# The line tells you what was the amount of variation in the value
# The line in the bar chart is call confidence interval meaning that 50% of the 
# values lies in the line

# In[98]:


#SEX
sns.barplot(x = 'day', y = 'total_bill', hue = 'sex', data = tips);


# In[101]:


## SMOKERS

sns.barplot(x = 'day', y = 'total_bill', hue = 'smoker', data = tips);


# In[107]:


#TIME 
sns.barplot(x = 'day', y = 'total_bill', hue ='time', data = tips);


# In[ ]:





# Heat map; It is used for a two dimentional data like a matrix or a table using colors.
# The best way to understand it is by looking at an example.
# using flight data set from seaborn to visualize monthly passengers footfall at an airport over 12yrs.

# In[114]:


import seaborn as sns


# In[122]:


flights = sns.load_dataset('flights')
flights


# looking at the trend

# In[139]:


plt.plot(flight.passengers);


# In[147]:


sns.barplot(flights);
plt.title('No of Passengers That Fly Yearly');


# To represent the data as a heatmap you have to first represent the data as a matrix. Using a pivot method on the data frame, and it will put the data in a kind of matrix format.

# In[135]:


flights = sns.load_dataset('flights')
flights = flights.pivot(index = 'month', columns='year', values='passengers')
flights


# The data above shows how passengers were flying monthly every year.

# In[149]:


sns.heatmap(flights);
plt.title('No of Passengers');


# Interpretation: The darker the number the less passengers and the lighter the numbers the higher the passengers that fly.
# Passengers tend to fly more in July and August.
# The no of passengers flying yeary tend to grow year by yyear.

# Styling the heat map

# In[159]:


sns.heatmap(flights, fmt = 'd', annot = True, cmap = 'Greens');
plt.title('No of Passengers');


# Images
# Matplotlib can also be used to display images.
# For image to be displayed, it has to be read into memory using PIL module

# In[171]:


from urllib.request import urlretrieve
from PIL import Image


# # Ploting Multiple Charts in a Grid
# 
# Matplotlib and seaborn supports multiple chars in a grid, using 
# plt.subplots
# which returns a set of axis that can be used for plotting.
# 

# In[172]:


fig, axes = plt.subplots(2, 3, figsize = (16, 8))


# In[263]:


fig, axes = plt.subplots(2, 3, figsize = (14, 10))  #making the subplots

plt.tight_layout(pad=3)   #space between each grid

#using the axis for plotting


#Axes 0 Crops 
axes[0, 0].plot(years, apples, 's-b');
axes[0, 0].plot(years, oranges, 's-r');
axes[0, 0].set_xlabel('Year')   # specific in matplotlip
axes[0, 0].set_ylabel('Yield (tons per hectare)');  #specific in matplotlib
axes[0, 0].legend(['Apples', 'Oranges']);
axes[0, 0].set_title('Crops Yield in Kanto');



#plot the axes to seaborn
sns.scatterplot(x='sepal length', y='sepal width', data = flowers,
               hue = 'species',
                s= 100, ax =axes[0,1]);
axes[0, 1].set_title('Sepal Length vs Width');


#the ax=axes[0, 1] is always specified inside sns 
#for matplotlib you can directly call the function directly e.g axes[0, 0].plot(years, apples, 's-b');
#plt.title('Sepal Length vs Width')  #sns is still correct.


#use the axis for ploting ---matplotlib
axes[0, 2].set_title('Distribution of Sepal Width')
axes[0, 2].hist([setosa['sepal width'], versicolor['sepal width'], virginica['sepal width']],
               bins = np.arange(2, 5, 0.25), stacked = True);

axes[0, 2].legend(['Setosa', 'Versicolor', 'Virginica']);



#plot the axes to seaborn

axes[1, 0].set_title('Restaurant Bills')
sns.barplot(x = 'day', y ='total_bill', hue = 'sex', data=tips, ax=axes[1, 0]);


#plot the axes into seaborn
axes[1, 1].set_title('Restaurant Tips')
sns.barplot(x = 'day', y ='tip', hue = 'time', data=tips, ax=axes[1, 1]);

#plot the axes into seaborn
axes[1, 2].set_title('Flight Traffic')
sns.heatmap(flights, cmap = 'Blues', fmt = 'd', annot = True, ax=axes[1,2]);





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




