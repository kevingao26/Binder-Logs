#!/usr/bin/env python
# coding: utf-8

# In[274]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

sns.set_style("whitegrid")
from pprint import pprint
import sqlite3
import yaml
import math
import re
from datetime import datetime
from datetime import timezone
import maya
from collections import Counter
from glob import glob
import ast
import sys
import json
import cProfile
import time
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from scipy import spatial

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

import networkx as nx
import networkx.algorithms.community as nxcom

import bokeh.io
from bokeh.io import output_file, show
from bokeh.resources import INLINE
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,
                          BoxZoomTool, ResetTool, OpenURL, CustomJS, Column, SaveTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_notebook
from bokeh.models.graphs import from_networkx
from bokeh.models import TextInput, Button

from scipy.spatial.distance import cosine
import scipy.stats as ss

# Code for hiding seaborn warnings
import warnings

warnings.filterwarnings("ignore")

# ## Easier navigation:
#
# ------------------------------------
#
# ### 1. <a href='#setup'>Setup</a>
# ### 2. <a href='#func'>Supporting Functions</a>
# ------------------------------------
#
# ## Models:
# ### 3. <a href='#naive'>Naive</a>
# ### 4. <a href='#lru'>Least Recently Used (LRU)</a>
# ### 5. <a href='#landlord'>Basic LANDLORD</a>
# ### 6. <a href='#landlordplus'>LANDLORD++</a>
# ------------------------------------
# ### 7. <a href='#runtime'>How Long to Run</a>
# ### 8. <a href='#results'>Results</a>

# <a name='setup'></a>
# ## 1. Setup
#
# <br>

# In[2]:


try:
    launches_df = pd.read_pickle("launches_with_versions.pkl")
    need_launch_stuff = False
except:
    con = sqlite3.connect('launches_spec.db')
    launches_df = pd.read_sql('SELECT * FROM chopped', con)
    need_launch_stuff = True

# In[241]:


with open('pentagon_df.txt') as f:
    reloaded_example = json.load(f)
q_df = pd.read_json(reloaded_example)

# In[242]:


dep_filtered = []
for index, row in q_df.iterrows():
    dep_filtered.append([p for p in row["dependencies full"] if p not in row["dependencies drained"]])

q_df["dependencies filtered"] = dep_filtered

# In[243]:


v_list = []
for index, row in q_df.iterrows():
    v_dict = {}
    try:
        for i in np.arange(len(row["Version Number"])):
            v_dict[row["Version Number"][i]] = row["Version Time"][i]
    except:
        pass

    v_list.append(v_dict)

q_df["Version Dict"] = v_list

# In[6]:


dep_df = pd.read_hdf('dependency_table_final.h5', 'df')
dep_df = dep_df.rename(columns=dep_df.loc["ref"]).drop("ref")

# ------------------------------------------------

# In[7]:


launches_df

# In[116]:


q_df.head()

# In[229]:


q_df.columns

# In[118]:


dep_df

# In[10]:


dep_list = q_df.index.to_list()
dep_list

# In[11]:


dep_binary_df = dep_df.copy(deep=True)
dep_binary_df = dep_binary_df.astype(bool).astype(int)
dep_binary_df

# Some libraries don't have version data.

# In[12]:


q_df[q_df["Version Dict"].apply(lambda x: len(x) == 0)]


# Convert launches_db's timestamps from string to a unix int to match the version time column from q_df.
# value represents milliseconds since unix date.

# In[13]:


def str2date2unix(s):
    dt = maya.parse(s).datetime()
    return dt.replace(tzinfo=timezone.utc).timestamp() * 1000


# In[14]:


def series2time_dict(se, timestamp):
    ret_dict = {}

    se_dict = se[se != 0].to_dict()
    filtered_dict = {k: se_dict[k] for k in dep_list if k in se_dict}

    for key, value in filtered_dict.items():
        lib_dict = q_df.at[key, "Version Dict"]
        above_time_dict = {k: v for (k, v) in lib_dict.items() if v <= timestamp}
        if len(above_time_dict) == 0:
            pass
        else:
            ret_dict[key] = max(above_time_dict, key=above_time_dict.get)

    return ret_dict


if need_launch_stuff:
    launch_version_dict = []
    count = 0
    for index, row in launches_df.iterrows():
        ref = row["combined_ref"]
        timestamp = str2date2unix(row["timestamp"])
        launch_version_dict.append(series2time_dict(dep_df[ref], timestamp))
        count += 1
        if count % 100000 == 0:
            print(count)
    launches_df["Package Versions"] = launch_version_dict
    launches_df = launches_df[["timestamp", "combined_ref", "Package Versions"]]
    launches_df[["timestamp", "combined_ref", "Package Versions"]].to_pickle("launches_with_versions.pkl")


# ------------------------------------------------

# <a name='func'></a>
# ## 2. Supporting Functions
#
# <br>

# Similarity between two recipes/columns:

# In[15]:


def cosine_sim(c1, c2):
    return 1 - cosine(c1, c2)


cosine_sim([1, 1, 1, 1], [1, 1, 0, 1])


# Combine two recipes/columns:

# In[16]:


def combine_col(c1, c2):
    return 1 - (1 - c1) * (1 - c2)


# Check if all of c1 is contained in c2:

# In[17]:


def contains_image(c1, c2):
    # Previous command that resulted in huge runtime increase
    # set(c1[c1 == 1].index)

    s = set(c1.to_numpy().nonzero()[0])
    image = set(c2.to_numpy().nonzero()[0])

    is_subset = s.issubset(image)
    return is_subset


# Container class:
#
# "pack" and "dep" both represent packages (Dependencies are included in the packages)

# In[421]:


class Container:

    # important ones are dep_version_numerical_dict, pack_list, df, size, time
    def __init__(self, dep_version_dict, versions, launch_count, xtra_vers,
                 xtra_stat1, xtra_dynamic, stat_version, heuristic_ct, xtra_size, xtra_time):

        self.dep_version_dict = dep_version_dict
        self.has_vers = xtra_vers
        self.has_stat1 = xtra_stat1
        self.has_dynamic = xtra_dynamic
        self.has_size = xtra_size
        self.has_time = xtra_time

        self.extra_vers = 0
        self.extra_stat1 = 0
        self.extra_dynamic = 0
        self.extra_size = 0
        self.extra_time = 0
        self.extra_score = 0

        self.stat_version = stat_version
        self.heuristic_ct = heuristic_ct

        self.refactor(versions, launch_count)

    # some parts that are needed in both init and when combining
    def refactor(self, versions, launch_count):

        self.pack_list = [*self.dep_version_dict]

        if self.has_dynamic:
            q_df.loc[self.pack_list, "Dynamic Freq"] += 1

        self.df = q_df.loc[self.pack_list]
        self.df_size = len(self.df)
        self.size = np.sum(self.df["size"])
        self.time = np.sum(self.df["time"])

        if self.has_vers:
            self.extra_vers = np.sum(self.df["Version Count"]) / self.df_size / max(self.df["Version Life"])

        if self.has_stat1:
            if self.stat_version == "a":
                self.extra_stat1 = np.sum(self.df["stat1"]) / self.df_size
            elif self.stat_version == "b":
                self.extra_stat1 = np.sum(self.df["stat1"]) / self.size

        if self.has_dynamic:
            self.extra_dynamic = np.sum(self.df["Dynamic Freq"]) / len(self.df) / launch_count

        if self.has_size:
            self.extra_size = self.size

        if self.has_time:
            self.extra_time = self.time

        if self.heuristic_ct == "A":
            if launch_count >= 100000:
                self.extra_score = self.has_vers * self.extra_vers / 0.3 + self.has_stat1 * self.extra_stat1 / 2100 + self.has_dynamic * self.extra_dynamic / 0.01
            else:
                self.extra_score = self.has_vers * self.extra_vers / 0.3 + self.has_stat1 * self.extra_stat1 / 2100 + 0.01
        elif self.heuristic_ct == "B":
            # cacheRank
            pass
        else:
            self.extra_score = self.extra_vers + self.extra_stat1 + self.extra_dynamic

        '''
        if self.p_switch:
            self.dep_version_dict = {k: v for k, v in self.dep_version_dict.items() if k in self.p_list}
            self.pack_list = [*self.dep_version_dict]

            self.df = self.p_df.loc[self.pack_list]
            temp_size = np.sum(self.p_df["size"])
            temp_time = np.sum(self.p_df["time"])
            self.excess_size = self.size - temp_size
            self.excess_time = self.time - temp_time

            self.size = temp_size
            self.time = temp_time
            '''

        '''
        # only picks keys(libraries) that are in dep_list
        self.dep_version_dict = self.filter_packages(self.testdict)
        '''

        '''
        if versions:
            # numerical uses a numerical time marker instead of version number
            self.dep_version_numerical_dict = {}
            for key in self.dep_version_dict:
                unix_val = version2unix(key, self.dep_version_dict[key], timestamp)
                if unix_val == "i":
                    self.dep_version_numerical_dict[key] = "before"
                elif unix_val == "e":
                    # replace
                    self.dep_version_numerical_dict[key] = "e"
                else:
                    self.dep_version_numerical_dict[key] = unix_val
        '''

        # update the dictionary with dependencies
        # self.pack_dict = self.add_dependencies_dict(self.dep_version_dict)
        # self.pack_dict = self.dep_version_dict

        '''
        if versions:
            # [*a] for a dictionary a gives a list of its keys
            self.pack_list = [*self.dep_version_numerical_dict]
        else:
            self.pack_list = [*self.dep_version_dict]
        '''

    # Edit: this part was done in another notebook to save computation time
    '''
    # Factor in dependencies of packages being added. The version of these added packages
    # will be labeled with "I".
    def add_dependencies_dict(self, pack_dict):

        # For dictionaries x and y, z becomes a shallowly merged 
        # dictionary with values from y replacing those from x.
        # z = {**x, **y}  // Python 3.5+
        # z = x | y // Python 3.9+
        # So y should be the original dict. x will take the keys/values from y's dependencies.

        new_dict = {}
        for pack in pack_dict:
            try:
                dep_full = q_df.at[pack, "dependencies filtered"]
                for dep in dep_full:
                    new_dict[dep] = "I"
            except:
                pass

        return {**new_dict, **pack_dict}

    # Input is a list instead of dictionary.
    def add_dependencies_list(self, pack_list):

        new_list = []
        for pack in pack_list:
            try:
                dep_full = q_df.at[pack, "dependencies filtered"]
                new_list.append(pack)
                for dep in dep_full:
                    new_list.append(dep)
            except:
                pass

        # will have duplicate elements in return, most likely (set needed later)
        return new_list
    '''

    # only picks keys(libraries) that are in dep_list
    def filter_packages(self, pack_dict):
        return {k: pack_dict[k] for k in dep_list if k in pack_dict}

    # checks if an image is fully captured by a container
    def compare_contains(self, other_dict, versions):
        if versions:
            # self.valid_intersection(other_dict, timestamp)
            if not self.version_compatible(self.dep_version_dict):
                return False
        return set(other_dict).issubset(set(self.dep_version_dict))

    # checks if an image is identical to its container
    def compare_identical(self, other_dict, versions):
        if versions:
            return other_dict == self.dep_version_dict
        return set(other_dict) == set(self.dep_version_dict)

    # combines an image with the container
    def combine(self, other, versions, launch_count):
        # starts from the LANDLORD model
        self.dep_version_dict = {**self.dep_version_dict, **other}
        self.refactor(versions, launch_count)

    # Not currently being used
    # determines if new lib shares same versions as container's libs
    def valid_intersection(self, o_dict, timestamp):
        other_dict = self.filter_packages(o_dict)
        self_set = set(self.dep_version_dict)
        other_set = set(other_dict)

        for lib in self_set.intersection(other_set):
            self_unix = self.dep_version_numerical_dict[lib]
            other_unix = version2unix(lib, other_dict[lib], timestamp)
            print(self_unix, other_unix)

    def version_compatible(self, other_dict):
        incompatible_size, incompatible_time = 0, 0
        for k in set(self.dep_version_dict).intersection(set(other_dict)):
            if self.dep_version_dict[k] != other_dict[k]:
                # print(k, self.dep_version_dict[k], other_dict[k])
                incompatible_size += q_df.at[k, "size"]
                incompatible_time += q_df.at[k, "time"]
        return incompatible_size, incompatible_time


# In[427]:


'''
a = {1:1, 2:2, 4:4}
b = {1:1, 2:2, 3:3}
for i in set(a).intersection(set(b)):
    print(a[i] == b[i])
'''

# In[428]:


'''
testct = nl_containers[4]
testct.add_dependencies_dict(testct.dep_version_dict)
testct.dep_version_dict
'''

# In[429]:


'''
a = {1:5, 2:7, 3:8}
b = {4:4, 1:3}
{**a, **b}
for i in a:
    print(i)
'''

# Note: a lot of launches won't be used - specs only has 34400 unique refs

# In[22]:


len(launches_df["combined_ref"].unique())


# Probably won't need this - checks if the launch is out of date.

# In[23]:


def check_timestamp(timestamp):
    # get min (doesn't consider "Default" values)
    # min_time_required = min([q_df.loc[i]["Version Dict"][xp[i]] for i in xp.to_dict()])

    min_val = sys.maxsize
    for i in xp.to_dict():
        try:
            val = q_df.loc[i]["Version Dict"][xp[i]]
        except:
            val = q_df.loc[i]["Version Time"][0]
        if val < min_val:
            min_val = val
            if min_val < timestamp:
                return False
    return min_val, timestamp


# Convert a library's version to a number for comparisons. Really useful!
#
# - "e" represents error, if the image explicitly called for a lib and it was out of date.
# - "i" represents that it was "I" (a package's dependency) before, so it can just be ignored.

# In[24]:


def version2unix(lib, version, timestamp):
    try:
        return q_df.loc[lib]["Version Dict"][version]
    except:
        test_dict = q_df.loc[lib]["Version Dict"]

        # temporary solution for libraries that don't have versions
        if len(test_dict) == 0:
            return "e"

        above_time_dict = {k: v for (k, v) in test_dict.items() if v <= timestamp}
        if above_time_dict == {}:
            if version == "I":
                return "i"
            else:
                # also temp
                print(lib, version, timestamp)
                return "e"
        return max(above_time_dict, key=above_time_dict.get)


# <a name='naive'></a>
# ## 3. Naive

# <a name='lru'></a>
# ## 4. Least Recently Used (LRU) - Identical / Contained

# <a name='landlord'></a>
# ## 5. LANDLORD
#
# <br>

# Non-Pythonic solution: An LRU cache is built by combining two data structures: a doubly linked list and a hash map. O(1) time by looking at the tail of the list, and O(1) time to access a specific element using the hashmap. All operations should be O(1) time, and O(n) space complexity.
#
# Python: OrderedDict, also O(1) in all operations, except for the newly added `get_total_size()`.
#
# Key will be some random unique index counting upwards.
# Value will be a container class object.
#
# https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/

# In[347]:


class LRUCache:

    def __init__(self, ct_limit: int, cache_limit, capacity: int, xtra, safe, heuristic_ct,
                 xtra_vers, xtra_stat1, xtra_dynamic, lru_combine, xtra_size, xtra_time):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ct_limit = ct_limit
        self.cache_limit = cache_limit
        self.extra = xtra
        self.safe = safe
        self.heuristic_ct = heuristic_ct
        self.mult_extra_vers = xtra_vers
        self.mult_extra_stat1 = xtra_stat1
        self.mult_extra_dynamic = xtra_dynamic
        self.lru_combine = lru_combine
        self.mult_extra_size = xtra_size
        self.mult_extra_time = xtra_time

    # cache.get(1)
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            return self.cache[key]

    # cache.shift(1)
    def shift(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    # cache.put(2, 2)
    def put(self, key: int, value: int):
        if value.size > self.ct_limit:
            return
        self.cache[key] = value
        self.cache.move_to_end(key)
        if (len(self.cache) > self.capacity) or (
                (self.cache_limit) and (self.get_total_size() > self.cache_limit)):
            if self.extra and (self.safe < self.capacity):
                self.remove_lowest()
            else:
                self.cache.popitem(last=False)

    def get_total_size(self):
        sizes, times = 0, 0
        for key in self.cache:
            ct = self.cache.get(key)
            sizes += ct.size
            # times += ct.time
        return sizes

    # cache.remove(1)
    def remove(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            self.cache.popitem(last=True)

    # for xtra set, removes cache item with lowest score (breaks ties with LRU)
    # cache loops from most recently used to lr used.
    # side note: if we want lru to be considered for 1 xtra being used then just the first if for everything
    def remove_lowest(self):
        temp_safe = self.safe
        lowest_score = 99999
        lowest_key = -1

        # cacheRank
        # A for dumb combine
        # B for basic cacherank
        # C for B with standardization

        temp_not_safe = self.capacity - self.safe

        raw_keys, raw_vers, raw_stat, raw_dyn = [], [], [], []
        raw_size, raw_time = [], []
        lru_rank = []

        lru_count = 1

        # start from least recently used so tiebreak is on least first
        for key in self.cache:
            if temp_not_safe > 0:
                temp_not_safe -= 1
                ct = self.cache.get(key)
                raw_keys.append(key)
                raw_vers.append(ct.extra_vers)
                raw_stat.append(ct.extra_stat1)
                raw_dyn.append(ct.extra_dynamic)
                raw_size.append(ct.extra_size)
                raw_time.append(ct.extra_time)
                lru_rank.append(lru_count)
                lru_count += 1

        if self.heuristic_ct == "A":
            a_vers = np.array(raw_vers)
            a_stat = np.array(raw_stat)
            a_dyn = np.array(raw_dyn)
            a_size = np.array(raw_size)
            a_time = np.array(raw_time)
        else:
            l_vers = ss.rankdata(raw_vers)
            l_stat = ss.rankdata(raw_stat)
            l_dyn = ss.rankdata(raw_dyn)
            l_size = ss.rankdata(raw_size)
            l_time = ss.rankdata(raw_time)

            a_vers = np.array(l_vers)
            a_stat = np.array(l_stat)
            a_dyn = np.array(l_dyn)
            a_size = np.array(l_size)
            a_time = np.array(l_time)

        def standardize_a(l):
            return (l - np.mean(l)) / np.std(l)

        if self.heuristic_ct == "C":
            a_vers = standardize_a(a_vers)
            a_stat = standardize_a(a_stat)
            a_dyn = a_dyn
            a_size = standardize_a(a_size)
            a_time = standardize_a(a_time)

        l_score = a_vers * self.mult_extra_vers + a_stat * self.mult_extra_stat1 + a_dyn * self.mult_extra_dynamic
        l_score += a_size * self.mult_extra_size + a_time * self.mult_extra_time

        if self.lru_combine:
            l_score += np.array(lru_rank)

        l_score = l_score.tolist()

        min_key = raw_keys[l_score.index(min(l_score))]

        self.remove(min_key)


# In[430]:


'''
cache = LRUCache(60000, 100000000, 2)

cache.put(1, nl_containers[0])
print(cache.cache)
cache.put(2, nl_containers[1])
print(cache.cache)
cache.put(3, nl_containers[2])
print(cache.cache)
cache.shift(2)
print(cache.cache)
cache.remove(2)
print(cache.cache)
'''

# Parameters:
#  - kind:
#      - "naive" - no caching at all. new container for each launch
#      - "lru_i" - only shares identical containers (same repo)
#      - "lru_c" - shares a launch if its image specification is fully covered by a container
#      - "land" - basic LANDLORD model. shares launch is fully covered or similar enough to a threshold alpha.
#  - custom: specifies how much of the dataset to use. Only set if testing, otherwise it defaults to None for the entire dataset.
#  - constraints:
#      - ct_size: max size allowed for a single container in the cache
#      - cache_size: max size allowed for all containers in the cache together
#      - capacity: how many to store in cache, default is unlimited
#  - versions: In the case of landlord, the "True" version (version is considered) does not follow the original model (?)
#      - "naive" - True / False
#      - "lru_i" - True / False
#      - "lru_c" - True / False
#      - "land" -   True / False
#  - alpha:
#      - parameter for "land"
#
#
# Concerns with LANDLORD:
#      - grouping of packages based on similarity is dependent on order. for example, if B passes the threshold to be merged with A, this might cause C and modified A to not be similar enough, whereas if C was compared first, C might be merged with A.
#      - packages that "conflict"
#
#
# HITRATE ONLY CALCULATED FOR RECIPES WITH >= 1 LIB RIGHT NOW
#
# hitrate = (total libs - # of containers created) / total libs

# Ideas:
#
# All of these are based on applying some sort of "filter" to the dataset of libraries, where we will only include libraries in a container if they pass this filter/conditions.
#
#  - Create a column `Version Change Avg` that keeps track of the average time between two updates for a library. If this time is below some threshold [a], then do not include this package in any containers (uninstall after use). If there are only [b] or less versions in total, then there is not enough data or the package is not updated enough and we assume it is ok -> this method only looks at times between updates, so if a package has 3 updates in 3 years and they all happened to be on the first day, then obviously we should use it (but the algo will say no). A solution to this would be to use the current time and subtract from the library's first release, which would work except we don't know which libraries went out of date and at what point.
#  - A metric such as, `Forks` / min(1, `Total Size`), will be applied on every library. If the library doesn't meet a certain threshold, then it will not be used in the container.
#  - To build on the previous one, we can dynamically record the proportion of times when each package has been called (and perhaps, take its version into account). Only use libraries that are above a certain proportion. Once again, this can somehow be scaled with `Total Size`, and also note that the first few (50?) launches will not create any containers because there is not enough data.

# Beyond Landlord:
#  - Version avg: In q_df, create a column for the total number of versions (minus one for the first one) for each library, and the timestamp of the very first version. Then, when a container is being created, look at the minimum of its library's timestamps, and subtract that from the date we scraped the sites to get the total amount of time in between `(July 1, 2021 - min(first version of all libs))`. Our metric is **version changes / time(UNIX time-milliseconds)**, and a threshold will be used to determine if the container passes the requirements.
#  - Popularity: We can take any combination of scraped Github statistics and make a column for this for each library. For example, stars * a + forks * b. Our metric is **sum(stat column for all libs) / sum(size for all libs)**, and a threshold will be used to determine if the container passes the requirements. These columns will be named **statx, where x is a number**.
#  - Dynamic: Keep a column that tracks the number of times a package was called in the launch, and update after every launch. The metric is **sum(new column for all libs) / number of libs / number of launches**. A threshold will be applied, like usual.
#
# If they pass, keep the container. Else, build a container but do not include it in the cache for Container(). Remove from cache after updating the container for combine().

# In[239]:


scrape_time = 1625122800000

# In[333]:


# time of first version for each library
first_version = q_df["Version Time"].apply(lambda x: x[-1] if ((x is not None) and (len(x) > 0)) else -1)
# number of days since first version
q_df["Version Life"] = (scrape_time - first_version) / (1000 * 60 * 60 * 24)

# number of versions
q_df["Version Count"] = q_df["Version Time"].apply(lambda x: len(x) if (x is not None) else 0)

# Sum of stars and forks divided by their medians
q_df["stat1"] = (q_df["Stars"] / q_df["Stars"].median()) + (q_df["Forks"] / q_df["Forks"].median())

# In[334]:


(q_df["Version Count"] / q_df["Version Life"])

# In[336]:


np.sum(q_df["Version Count"]) / min(q_df["Version Life"])

# In[326]:


q_df["stat1"]


# In[431]:


def MODEL_nl(kind, custom=None, constraints=[-1, -1, -1], versions=True, alpha=0.7, xtra_vers=0, xtra_stat1=0,
             xtra_dynamic=0, stat_version="", heuristic_ct="", heuristic_land="", cache_safe=0, lru_combine=True,
             xtra_size=0, xtra_time=0):
    '''
    if vers_alpha:
        # Average time between repo versions for a repo. There are two values that can be changed/optimized.
        q_df["Version Change Avg"] = q_df["Version Time"].apply(
            lambda x: ((x[0] - x[-1]) / (len(x) - 1) / (1000*60*60*24)) if ((
                x is not None) and len(x) > vers_alpha[0]) else -1)
        p_df = p_df[p_df["Version Change Avg"] > vers_alpha[1]]

    # Index to list.
    p_list = p_df.index.values.tolist()

    # Makes a true / false column based on if q_df index is in the filtered list of libs.
    q_df["Passed Filter"] = q_df.index.isin(p_list)
    '''

    xtra = xtra_vers or xtra_stat1 or xtra_dynamic

    xtra_count = 0
    if xtra_vers:
        xtra_count += 1
    if xtra_stat1:
        xtra_count += 1
    else:
        stat_version = ""
    if xtra_dynamic:
        xtra_count += 1

    if xtra_count < 2:
        heuristic_ct = ""
    if xtra_count < 1:
        heuristic_land = ""

    q_df["Dynamic Freq"] = 0

    if kind not in ["naive", "lru_i", "lru_c", "land"]:
        raise ValueError('Model type not specified correctly.')

    total_size, total_time = 0, 0

    constraints = [sys.maxsize if i == -1 else i for i in constraints]
    ct_limit, cache_limit, capacity = constraints
    if cache_limit == sys.maxsize:
        cache_limit = False

    start = time.time()
    cache = LRUCache(ct_limit, cache_limit, capacity, xtra, cache_safe, heuristic_ct,
                     xtra_vers, xtra_stat1, xtra_dynamic, lru_combine, xtra_size, xtra_time)
    count = 0
    ct_count = 0
    containers = []

    if kind != "land":
        heuristic_land = ""

    # version diff in matplotlib between 7714 and 7719

    if custom == "timeit":
        df = launches_df[0:2000]
    elif custom == "big":
        df = launches_df[0:500000]
    elif custom == "tests":
        df = launches_df[1200000:2011411]
    elif custom:
        df = launches_df[0:200]
    else:
        df = launches_df

    counter = 0
    for index, row in df.iterrows():
        counter += 1
        if counter % 100000 == 0:
            print(count)

        need_container = True
        ref = row["combined_ref"]
        timestamp = str2date2unix(row["timestamp"])

        df_versions_dict = df.at[index, "Package Versions"]

        '''
        dep_series = dep_df[ref][dep_df[ref] != 0]
        dep_list_m = dep_series.index
        '''

        # Reuse image specifications that are fully contained by a container
        if kind != "naive":
            # want reversed because we want to look at newest ones first (higher hit rate)
            for key in reversed(cache.cache):
                ct = cache.get(key)
                if (kind == "lru_c") or (kind == "land"):
                    if ct.compare_contains(df_versions_dict, versions):
                        cache.shift(key)
                        need_container = False
                        break
                elif (kind == "lru_i"):
                    if ct.compare_identical(df_versions_dict, versions):
                        cache.shift(key)
                        need_container = False
                        break

        if (kind == "land") and need_container:
            for key in reversed(cache.cache):
                ct = cache.get(key)
                ct_series = dep_df.index.map(lambda x: x in ct.pack_list).astype(int)
                if cosine_sim(dep_binary_df[ref], ct_series) > alpha:
                    if versions:
                        s, t = ct.version_compatible(df_versions_dict)
                        if s:
                            total_size += s
                            total_time += t

                    # combine container
                    ct.combine(df_versions_dict, versions, counter)
                    cache.shift(key)
                    need_container = False
                    break

        count += 1
        if df_versions_dict == {}:
            need_container = False
        if need_container:
            ct_count += 1
            new_ct = Container(df_versions_dict, versions, counter, xtra_vers, xtra_stat1, xtra_dynamic,
                               stat_version, heuristic_ct, xtra_size, xtra_time)
            containers.append(new_ct)
            if kind != "naive":
                # cache key will be ct_count(arbitrary) and value will be container
                cache.put(ct_count, new_ct)

    total_size += np.sum([ct.size for ct in containers])
    total_time += np.sum([ct.time for ct in containers])

    hitrate = 0
    if kind != "naive":
        hitrate = (count - ct_count) / count

    end = time.time()
    time_taken = end - start
    return time_taken, count, ct_count, containers, total_size, total_time, hitrate, cache, heuristic_ct, heuristic_land, stat_version
    return time_taken, count, containers, 1, 1


def to_dataframe(l1, l2, hue):
    return pd.DataFrame(
        {'x': l1,
         'y': l2,
         'hue': hue
         })


def to_dataframe(l1, l2, l3, hue):
    return pd.DataFrame(
        {'x': l1,
         'y': l2,
         'z': l3,
         'hue': hue
         })


def RESULT_model_nl(models, custom, version, constraints, alpha, xtra_vers, xtra_stat1, xtra_dynamic, stat_version,
                    heuristic_ct, heuristic_land, cache_safe, lru_combine, xtra_size, xtra_time):
    if models == "all":
        models = ["naive", "lru_i", "lru_c", "land"]
    elif models == "set":
        models = ["lru_i", "lru_c", "land"]

    total_size, total_time, hitrate, cache_size, names, timeit = [], [], [], [], [], []

    for model in models:
        start = time.time()

        temp = MODEL_nl(model, custom, constraints, version, alpha, xtra_vers, xtra_stat1, xtra_dynamic, lru_combine,
                        xtra_size, xtra_time)

        total_size.append(temp[4])
        total_time.append(temp[5])
        hitrate.append(temp[6])
        cache_size.append(temp[7].get_total_size())

        end = time.time()
        timeit.append(end - start)

        names.append(model)

    return models, custom, constraints, version, alpha, xtra_vers, xtra_stat1, xtra_dynamic, total_size, total_time, hitrate, cache_size, names, timeit, stat_version, heuristic_ct, heuristic_land, cache_safe, lru_combine, xtra_size, xtra_time


for j in [5, 10, 15, 25, 35, 50]:
    for k in [0, j // 5, 2 * j // 5, 3 * j // 5, 4 * j // 5]:
        results = RESULT_model_nl(["lru_c"], "tests", True, [-1, -1, j], 0.7, 0, 1, 0,
                                  "a", "C", "A", k, True, 1, 0)

        outF = open("model_results.txt", "a")
        for r in results:
            # write line to output file
            outF.write(str(r))
            outF.write("\n")

        outF.write("\n")
        outF.write("----------------------------")
        outF.write("\n")
        outF.close()

        print(j, k)

for j in [5, 10, 15, 25, 35, 50]:
    for k in [0, j // 5, 2 * j // 5, 3 * j // 5, 4 * j // 5]:
        results = RESULT_model_nl(["lru_c"], "tests", True, [-1, -1, j], 0.7, 0, 1, 0,
                                  "b", "C", "A", k, True, 0, 1)

        outF = open("model_results.txt", "a")
        for r in results:
            # write line to output file
            outF.write(str(r))
            outF.write("\n")

        outF.write("\n")
        outF.write("----------------------------")
        outF.write("\n")
        outF.close()

        print(j, k)

for j in [5, 10, 15, 25, 35, 50]:
    for k in [0, j // 5, 2 * j // 5, 3 * j // 5, 4 * j // 5]:
        results = RESULT_model_nl(["lru_c"], "tests", True, [-1, -1, j], 0.7, 0, 0, 1,
                                  "b", "C", "A", k, True, 0, 0)

        outF = open("model_results.txt", "a")
        for r in results:
            # write line to output file
            outF.write(str(r))
            outF.write("\n")

        outF.write("\n")
        outF.write("----------------------------")
        outF.write("\n")
        outF.close()

        print(j, k)

'''
for ct_size in np.arange(50,800,50):
    results = RESULT_model_nl(["lru_c"], "tests", True, [ct_size, -1, -1], 0.7, 1, 0, 0)

    outF = open("model_results.txt", "a")
    for i in results:
      # write line to output file
      outF.write(str(i))
      outF.write("\n")

    outF.write("\n")
    outF.write("----------------------------")
    outF.write("\n")
    outF.close()

    print(ct_size)
'''


