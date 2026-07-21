{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3e055dd7-441d-40aa-a49a-e2fd8bd0c5af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7296c5d9-85d7-4f45-87dd-8fda7547d7bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler,MinMaxScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import  LinearRegression,Lasso,Ridge\n",
    "from sklearn.metrics import  mean_absolute_error,mean_squared_error,r2_score\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.neighbors import KNeighborsRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "062ed2e2-545c-4011-a94e-58a6c2117b3b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>distance_to_solar_noon</th>\n",
       "      <th>temperature</th>\n",
       "      <th>wind_direction</th>\n",
       "      <th>wind_speed</th>\n",
       "      <th>sky_cover</th>\n",
       "      <th>visibility</th>\n",
       "      <th>humidity</th>\n",
       "      <th>average_wind_speed</th>\n",
       "      <th>average_pressure</th>\n",
       "      <th>power_generated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.859897</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>75</td>\n",
       "      <td>8.0</td>\n",
       "      <td>29.82</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.628535</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>77</td>\n",
       "      <td>5.0</td>\n",
       "      <td>29.85</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.397172</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>70</td>\n",
       "      <td>0.0</td>\n",
       "      <td>29.89</td>\n",
       "      <td>5418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.165810</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>33</td>\n",
       "      <td>0.0</td>\n",
       "      <td>29.91</td>\n",
       "      <td>25477</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.065553</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>21</td>\n",
       "      <td>3.0</td>\n",
       "      <td>29.89</td>\n",
       "      <td>30069</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2915</th>\n",
       "      <td>0.166453</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>4</td>\n",
       "      <td>10.0</td>\n",
       "      <td>75</td>\n",
       "      <td>10.0</td>\n",
       "      <td>29.93</td>\n",
       "      <td>6995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2916</th>\n",
       "      <td>0.064020</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>66</td>\n",
       "      <td>15.0</td>\n",
       "      <td>29.91</td>\n",
       "      <td>29490</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2917</th>\n",
       "      <td>0.294494</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>2</td>\n",
       "      <td>10.0</td>\n",
       "      <td>68</td>\n",
       "      <td>21.0</td>\n",
       "      <td>29.88</td>\n",
       "      <td>17257</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2918</th>\n",
       "      <td>0.524968</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>2</td>\n",
       "      <td>10.0</td>\n",
       "      <td>81</td>\n",
       "      <td>17.0</td>\n",
       "      <td>29.87</td>\n",
       "      <td>677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2919</th>\n",
       "      <td>0.755442</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>81</td>\n",
       "      <td>11.0</td>\n",
       "      <td>29.90</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2920 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      distance_to_solar_noon  temperature  wind_direction  wind_speed  \\\n",
       "0                   0.859897           69              28         7.5   \n",
       "1                   0.628535           69              28         7.5   \n",
       "2                   0.397172           69              28         7.5   \n",
       "3                   0.165810           69              28         7.5   \n",
       "4                   0.065553           69              28         7.5   \n",
       "...                      ...          ...             ...         ...   \n",
       "2915                0.166453           63              27        13.9   \n",
       "2916                0.064020           63              27        13.9   \n",
       "2917                0.294494           63              27        13.9   \n",
       "2918                0.524968           63              27        13.9   \n",
       "2919                0.755442           63              27        13.9   \n",
       "\n",
       "      sky_cover  visibility  humidity  average_wind_speed  average_pressure  \\\n",
       "0             0        10.0        75                 8.0             29.82   \n",
       "1             0        10.0        77                 5.0             29.85   \n",
       "2             0        10.0        70                 0.0             29.89   \n",
       "3             0        10.0        33                 0.0             29.91   \n",
       "4             0        10.0        21                 3.0             29.89   \n",
       "...         ...         ...       ...                 ...               ...   \n",
       "2915          4        10.0        75                10.0             29.93   \n",
       "2916          1        10.0        66                15.0             29.91   \n",
       "2917          2        10.0        68                21.0             29.88   \n",
       "2918          2        10.0        81                17.0             29.87   \n",
       "2919          1        10.0        81                11.0             29.90   \n",
       "\n",
       "      power_generated  \n",
       "0                   0  \n",
       "1                   0  \n",
       "2                5418  \n",
       "3               25477  \n",
       "4               30069  \n",
       "...               ...  \n",
       "2915             6995  \n",
       "2916            29490  \n",
       "2917            17257  \n",
       "2918              677  \n",
       "2919                0  \n",
       "\n",
       "[2920 rows x 10 columns]"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"solarpowergeneration.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "dc8cd3b2-edfd-47c4-be0b-c7e50c79c77c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 2920 entries, 0 to 2919\n",
      "Data columns (total 10 columns):\n",
      " #   Column                  Non-Null Count  Dtype  \n",
      "---  ------                  --------------  -----  \n",
      " 0   distance_to_solar_noon  2920 non-null   float64\n",
      " 1   temperature             2920 non-null   int64  \n",
      " 2   wind_direction          2920 non-null   int64  \n",
      " 3   wind_speed              2920 non-null   float64\n",
      " 4   sky_cover               2920 non-null   int64  \n",
      " 5   visibility              2920 non-null   float64\n",
      " 6   humidity                2920 non-null   int64  \n",
      " 7   average_wind_speed      2919 non-null   float64\n",
      " 8   average_pressure        2920 non-null   float64\n",
      " 9   power_generated         2920 non-null   int64  \n",
      "dtypes: float64(5), int64(5)\n",
      "memory usage: 228.3 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "4db80f6d-3680-4c7d-aa91-9eaa81c3fb30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>distance_to_solar_noon</th>\n",
       "      <th>temperature</th>\n",
       "      <th>wind_direction</th>\n",
       "      <th>wind_speed</th>\n",
       "      <th>sky_cover</th>\n",
       "      <th>visibility</th>\n",
       "      <th>humidity</th>\n",
       "      <th>average_wind_speed</th>\n",
       "      <th>average_pressure</th>\n",
       "      <th>power_generated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2919.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "      <td>2920.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.503294</td>\n",
       "      <td>58.468493</td>\n",
       "      <td>24.953425</td>\n",
       "      <td>10.096986</td>\n",
       "      <td>1.987671</td>\n",
       "      <td>9.557705</td>\n",
       "      <td>73.513699</td>\n",
       "      <td>10.129154</td>\n",
       "      <td>30.017760</td>\n",
       "      <td>6979.846233</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.298024</td>\n",
       "      <td>6.841200</td>\n",
       "      <td>6.915178</td>\n",
       "      <td>4.838185</td>\n",
       "      <td>1.411978</td>\n",
       "      <td>1.383884</td>\n",
       "      <td>15.077139</td>\n",
       "      <td>7.261547</td>\n",
       "      <td>0.142006</td>\n",
       "      <td>10312.336413</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.050401</td>\n",
       "      <td>42.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>29.480000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.243714</td>\n",
       "      <td>53.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>6.600000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>65.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>29.920000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.478957</td>\n",
       "      <td>59.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>77.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>404.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.739528</td>\n",
       "      <td>63.000000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>13.100000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>84.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>30.110000</td>\n",
       "      <td>12723.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.141361</td>\n",
       "      <td>78.000000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>26.600000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>30.530000</td>\n",
       "      <td>36580.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       distance_to_solar_noon  temperature  wind_direction   wind_speed  \\\n",
       "count             2920.000000  2920.000000     2920.000000  2920.000000   \n",
       "mean                 0.503294    58.468493       24.953425    10.096986   \n",
       "std                  0.298024     6.841200        6.915178     4.838185   \n",
       "min                  0.050401    42.000000        1.000000     1.100000   \n",
       "25%                  0.243714    53.000000       25.000000     6.600000   \n",
       "50%                  0.478957    59.000000       27.000000    10.000000   \n",
       "75%                  0.739528    63.000000       29.000000    13.100000   \n",
       "max                  1.141361    78.000000       36.000000    26.600000   \n",
       "\n",
       "         sky_cover   visibility     humidity  average_wind_speed  \\\n",
       "count  2920.000000  2920.000000  2920.000000         2919.000000   \n",
       "mean      1.987671     9.557705    73.513699           10.129154   \n",
       "std       1.411978     1.383884    15.077139            7.261547   \n",
       "min       0.000000     0.000000    14.000000            0.000000   \n",
       "25%       1.000000    10.000000    65.000000            5.000000   \n",
       "50%       2.000000    10.000000    77.000000            9.000000   \n",
       "75%       3.000000    10.000000    84.000000           15.000000   \n",
       "max       4.000000    10.000000   100.000000           40.000000   \n",
       "\n",
       "       average_pressure  power_generated  \n",
       "count       2920.000000      2920.000000  \n",
       "mean          30.017760      6979.846233  \n",
       "std            0.142006     10312.336413  \n",
       "min           29.480000         0.000000  \n",
       "25%           29.920000         0.000000  \n",
       "50%           30.000000       404.000000  \n",
       "75%           30.110000     12723.500000  \n",
       "max           30.530000     36580.000000  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe(include=\"all\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "4e42bf64-de27-458c-b864-14bf29afa4f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_to_solar_noon    0\n",
       "temperature               0\n",
       "wind_direction            0\n",
       "wind_speed                0\n",
       "sky_cover                 0\n",
       "visibility                0\n",
       "humidity                  0\n",
       "average_wind_speed        1\n",
       "average_pressure          0\n",
       "power_generated           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae1188dc-17a9-4cc3-b9d9-efedff531c14",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "40094328-5308-4d1e-b236-7e081f2dfdb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>distance_to_solar_noon</th>\n",
       "      <th>temperature</th>\n",
       "      <th>wind_direction</th>\n",
       "      <th>wind_speed</th>\n",
       "      <th>sky_cover</th>\n",
       "      <th>visibility</th>\n",
       "      <th>humidity</th>\n",
       "      <th>average_wind_speed</th>\n",
       "      <th>average_pressure</th>\n",
       "      <th>power_generated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.859897</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>75</td>\n",
       "      <td>8.0</td>\n",
       "      <td>29.82</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.628535</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>77</td>\n",
       "      <td>5.0</td>\n",
       "      <td>29.85</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.397172</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>70</td>\n",
       "      <td>0.0</td>\n",
       "      <td>29.89</td>\n",
       "      <td>5418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.165810</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>33</td>\n",
       "      <td>0.0</td>\n",
       "      <td>29.91</td>\n",
       "      <td>25477</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.065553</td>\n",
       "      <td>69</td>\n",
       "      <td>28</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>21</td>\n",
       "      <td>3.0</td>\n",
       "      <td>29.89</td>\n",
       "      <td>30069</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2915</th>\n",
       "      <td>0.166453</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>4</td>\n",
       "      <td>10.0</td>\n",
       "      <td>75</td>\n",
       "      <td>10.0</td>\n",
       "      <td>29.93</td>\n",
       "      <td>6995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2916</th>\n",
       "      <td>0.064020</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>66</td>\n",
       "      <td>15.0</td>\n",
       "      <td>29.91</td>\n",
       "      <td>29490</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2917</th>\n",
       "      <td>0.294494</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>2</td>\n",
       "      <td>10.0</td>\n",
       "      <td>68</td>\n",
       "      <td>21.0</td>\n",
       "      <td>29.88</td>\n",
       "      <td>17257</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2918</th>\n",
       "      <td>0.524968</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>2</td>\n",
       "      <td>10.0</td>\n",
       "      <td>81</td>\n",
       "      <td>17.0</td>\n",
       "      <td>29.87</td>\n",
       "      <td>677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2919</th>\n",
       "      <td>0.755442</td>\n",
       "      <td>63</td>\n",
       "      <td>27</td>\n",
       "      <td>13.9</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>81</td>\n",
       "      <td>11.0</td>\n",
       "      <td>29.90</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2919 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      distance_to_solar_noon  temperature  wind_direction  wind_speed  \\\n",
       "0                   0.859897           69              28         7.5   \n",
       "1                   0.628535           69              28         7.5   \n",
       "2                   0.397172           69              28         7.5   \n",
       "3                   0.165810           69              28         7.5   \n",
       "4                   0.065553           69              28         7.5   \n",
       "...                      ...          ...             ...         ...   \n",
       "2915                0.166453           63              27        13.9   \n",
       "2916                0.064020           63              27        13.9   \n",
       "2917                0.294494           63              27        13.9   \n",
       "2918                0.524968           63              27        13.9   \n",
       "2919                0.755442           63              27        13.9   \n",
       "\n",
       "      sky_cover  visibility  humidity  average_wind_speed  average_pressure  \\\n",
       "0             0        10.0        75                 8.0             29.82   \n",
       "1             0        10.0        77                 5.0             29.85   \n",
       "2             0        10.0        70                 0.0             29.89   \n",
       "3             0        10.0        33                 0.0             29.91   \n",
       "4             0        10.0        21                 3.0             29.89   \n",
       "...         ...         ...       ...                 ...               ...   \n",
       "2915          4        10.0        75                10.0             29.93   \n",
       "2916          1        10.0        66                15.0             29.91   \n",
       "2917          2        10.0        68                21.0             29.88   \n",
       "2918          2        10.0        81                17.0             29.87   \n",
       "2919          1        10.0        81                11.0             29.90   \n",
       "\n",
       "      power_generated  \n",
       "0                   0  \n",
       "1                   0  \n",
       "2                5418  \n",
       "3               25477  \n",
       "4               30069  \n",
       "...               ...  \n",
       "2915             6995  \n",
       "2916            29490  \n",
       "2917            17257  \n",
       "2918              677  \n",
       "2919                0  \n",
       "\n",
       "[2919 rows x 10 columns]"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.dropna(subset=[\"average_wind_speed\"])\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "68990841-40d2-41b5-80af-e2ce2b65daa8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['distance_to_solar_noon', 'temperature', 'wind_direction', 'wind_speed',\n",
      "       'sky_cover', 'visibility', 'humidity', 'average_wind_speed',\n",
      "       'average_pressure', 'power_generated'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "7a70ffbb-bdd4-4b5e-93c7-b8eb1c7345c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns=[\"wind_direction\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "18b85211-c21c-46c2-b826-3c0245c402b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_to_solar_noon    0.211104\n",
       "temperature               0.125680\n",
       "wind_speed                0.416894\n",
       "sky_cover                 0.080279\n",
       "visibility               -3.875780\n",
       "humidity                 -0.956141\n",
       "average_wind_speed        0.622910\n",
       "average_pressure          0.442221\n",
       "power_generated           1.306524\n",
       "dtype: float64"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "6da668df-9a78-44c1-936a-6af0727f5fd8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgQAAAEnCAYAAADW0luTAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAJeFJREFUeJzt3X1clHW+//H3cDeACi2gIIaA5s3kXTKspcbxrnCx7LRnS0+uoalnxZvDQykr8lTK1rKZmW0p2KZxTOtQm6fNXdaVzVJb2VYRjyfFtd1EKEGCCrxFge/vD3/MaQSBAWykXs/H43r4mC/f71yf65qBefu9bsZijDECAADfax7uLgAAALgfgQAAABAIAAAAgQAAAIhAAAAARCAAAAAiEAAAABEIAACACAQAAEAEArRSVlaWLBaL09K9e3eNHTtWv/vd79xdnkNUVJRmzpzp8rizZ89q2bJl+uCDDzq8puZcXu8HH3wgi8Xich1r165VVlZWh9Z2rXj99de1evVqd5fRajNnzlRUVJS7ywBcRiCAS1599VXl5eVpz549evnll+Xp6anJkydr69at7i6tXc6ePavly5d/64HgcjExMcrLy1NMTIxL4wgEANrLy90FoHMZPHiwYmNjHY9/9KMf6Qc/+IHeeOMNTZ482Y2VfTcEBATolltucXcZuIacPXtW/v7+7i4D3wPMEKBdfH195ePjI29vb6f2L7/8UvPnz1evXr3k4+OjPn36aOnSpaqpqZEknT9/XsOHD9cNN9ygqqoqx7iysjKFhYVp7Nixqqurk3RpCrZr1646dOiQJkyYoC5duqh79+5auHChzp4922KNxcXFmj59unr06CGr1SqbzabnnntO9fX1kqSioiJ1795dkrR8+XLHIZG2HHq4kosXL+rhhx9WWFiY/P39deutt+qvf/1ro35NHTL49NNP9a//+q8KDw+X1WpVaGioJkyYoAMHDki6dNjh0KFD2rlzp6P2hinr8+fP68EHH9RNN92kwMBABQUFaeTIkfrtb3/baN0Wi0ULFy7Ua6+9JpvNJn9/fw0bNqzJQ0JHjhzRfffdp9DQUFmtVvXu3VuJiYmO11e69FrOnTtX119/vXx8fBQdHa3ly5ertra21ftt7Nix+v3vf6/jx487Ha5q0NL7rLVa2seSVF9frxUrVmjgwIGyWq3q0aOHEhMT9dlnn7X4/GvWrNE//dM/qUePHurSpYuGDBmiFStW6OLFi422d/Dgwdq1a5dGjRolf39/zZo1q1XbsGzZMlksFh06dEj33XefAgMDFRoaqlmzZjn9jkmX3hepqamKjo6Wj4+PevXqpQULFujrr7926tfabW6oe+/evYqLi5O/v7/69OmjX/7yl47fM3QCBmiFV1991Ugyf/nLX8zFixfNhQsXTElJiUlOTjYeHh5m27Ztjr7nzp0zQ4cONV26dDErV64027dvN48//rjx8vIykyZNcvQ7evSo6datm/mXf/kXY4wxdXV1Zvz48aZHjx7mxIkTjn4zZswwPj4+pnfv3ubpp58227dvN8uWLTNeXl7mzjvvdKozMjLSzJgxw/G4vLzc9OrVy3Tv3t1kZmaabdu2mYULFxpJZt68ecYYY86fP2+2bdtmJJnZs2ebvLw8k5eXZ/7+97932P6bMWOGsVgsZsmSJWb79u1m1apVplevXiYgIMCp3vfff99IMu+//76jbcCAAeaGG24wr732mtm5c6d5++23zYMPPujos3//ftOnTx8zfPhwR+379+83xhjz9ddfm5kzZ5rXXnvN7Nixw2zbts089NBDxsPDw/znf/6nU42STFRUlBkxYoR58803TU5Ojhk7dqzx8vIy//jHPxz9Dhw4YLp27WqioqJMZmamee+998ymTZvMlClTTHV1tTHGmNLSUhMREWEiIyPNunXrzJ/+9Cfz85//3FitVjNz5sxW77dDhw6Z0aNHm7CwMMe25eXlGWNa/z5rjZb2sTHG/OxnPzOSzMKFC822bdtMZmam6d69u4mIiDBffPGFo9+MGTNMZGSk0/MvXrzYZGRkmG3btpkdO3aY559/3oSEhJgHHnjAqd+YMWNMUFCQiYiIMC+++KJ5//33zc6dO1u1DU8++aSRZAYMGGCeeOIJk5uba1atWmWsVqvTeurr683EiRONl5eXefzxx8327dvNypUrTZcuXczw4cPN+fPnXd7mMWPGmODgYNOvXz+TmZlpcnNzzfz5842kRu8zXLsIBGiVhkBw+WK1Ws3atWud+mZmZhpJ5s0333Rqf+aZZ4wks337dkdbdna2kWRWr15tnnjiCePh4eH0c2Mu/YGVZF544QWn9qefftpIMh9++KGj7fJA8OijjxpJ5qOPPnIaO2/ePGOxWMzf/vY3Y4wxX3zxhZFknnzySZf3TUsKCwuNJLN48WKn9s2bNxtJzQaCiooKx/5pzqBBg8yYMWNarKW2ttZcvHjRzJ492wwfPtzpZ5JMaGio40PdGGPKysqMh4eHSU9Pd7SNHz/eXHfddaa8vPyK65k7d67p2rWrOX78uFP7ypUrjSRz6NChFmttcMcddzT6gDXGtfdZc1qzjxtew/nz5zu1f/TRR0aSeeyxxxxtTQWCb6qrqzMXL140GzduNJ6enubLL790/GzMmDFGknnvvfdaVfs3NQSCFStWOLXPnz/f+Pr6mvr6emOMcYTfy/s1/C6+/PLLLm9zQ92X/57deOONZuLEiS5vC9yDQwZwycaNG7V3717t3btXf/jDHzRjxgwtWLBAL730kqPPjh071KVLF91zzz1OYxum4N977z1H25QpUzRv3jwtWbJETz31lB577DHdfvvtTa77pz/9qdPjadOmSZLef//9K9a7Y8cO3XjjjRoxYkSjWowx2rFjR8sb3YS6ujrV1tY6luamRRvqu7z+KVOmyMur+dN4goKC1LdvXz377LNatWqVCgoKXJ6CfeuttzR69Gh17dpVXl5e8vb21vr161VYWNio77hx49StWzfH49DQUPXo0UPHjx+XdOl49s6dOzVlyhTHYZam/O53v9O4ceMUHh7utJ8SEhIkSTt37nRpG5riyvusOa3Zxw2v4eWHkUaMGCGbzdbiugoKCnTXXXcpODhYnp6e8vb2VmJiourq6nT06FGnvj/4wQ80fvz4VtXelLvuusvp8dChQ3X+/HmVl5dLkuM9f/m23HvvverSpYtjW1zd5rCwsEa/Z0OHDnW8d3DtIxDAJTabTbGxsYqNjdWPfvQjrVu3TvHx8Xr44Ycdxx8rKysVFhbmdKxXknr06CEvLy9VVlY6tc+aNUsXL16Ul5eXkpOTm1yvl5eXgoODndrCwsIc67uSyspK9ezZs1F7eHh4i2Ob07dvX3l7ezuWtLS0Zmv4Zr0Nmtqmy1ksFr333nuaOHGiVqxYoZiYGHXv3l3Jyck6depUi3Vu2bJFU6ZMUa9evbRp0ybl5eVp7969mjVrls6fP9+of1P1WK1WnTt3TpL01Vdfqa6uTtdff32z6z158qS2bt3qtI+8vb01aNAgSVJFRUWLtbfE1ffZlbRmHzc815XeS82tq7i4WHFxcfr888/1wgsvaPfu3dq7d6/WrFkjSY5926Cpdbji8tfQarU6raeyslJeXl6NAp3FYlFYWJhjW1zd5pbeO7j2cZUB2m3o0KH64x//qKNHj2rEiBEKDg7WRx99JGOM0x/r8vJy1dbWKiQkxNF25swZ3X///erfv79OnjypOXPmNHnCW21trSorK53+6JSVlUlq+g9Rg+DgYJWWljZqP3HihCQ51eKKrVu3Op241hAwrlRDQ729evVytDdsU0siIyO1fv16SdLRo0f15ptvatmyZbpw4YIyMzObHbtp0yZFR0crOzvb6bVw9aS7BkFBQfL09GzxRLqQkBANHTpUTz/9dJM/b25/tZYr77OWtLSPG17D0tLSRmHoxIkTza7rnXfe0ZkzZ7RlyxZFRkY62r95wuI3XR5wOlpwcLBqa2v1xRdfOIUCY4zKysr0wx/+0NFPats2o3NihgDt1vCHreGPy4QJE3T69Gm98847Tv02btzo+HmDpKQkFRcXa8uWLVq/fr3effddPf/8802uZ/PmzU6PX3/9dUmXznC+kgkTJujw4cPav39/o1osFovGjRsnqfH/oloyZMgQx0xJbGxssx9wDfVdXv+bb77p0hn3ktS/f3/9x3/8h4YMGeK0TVf6n5jFYpGPj4/Th0xZWVmToas1/Pz8NGbMGL311lvN/i//zjvv1Mcff6y+ffs67afW7K/LXWnbXHmfuaKpfdwwhb9p0yanvnv37lVhYWGz62rY9w3vMenSh++vf/3rNtXXXg21Xr4tb7/9ts6cOeP4eXu2GZ0TMwRwyccff+z4EKusrNSWLVuUm5urH//4x4qOjpYkJSYmas2aNZoxY4aKioo0ZMgQffjhh/rFL36hSZMm6bbbbpMkvfLKK9q0aZNeffVVDRo0SIMGDdLChQv1yCOPaPTo0U7HI318fPTcc8/p9OnT+uEPf6g9e/boqaeeUkJCgm699dYr1rt48WJt3LhRd9xxh9LS0hQZGanf//73Wrt2rebNm6f+/ftLkrp166bIyEj99re/1YQJExQUFKSQkJAOueOczWbT9OnTtXr1anl7e+u2227Txx9/rJUrVyogIKDZsQcPHtTChQt17733ql+/fvLx8dGOHTt08OBBPfroo45+Q4YM0X/9138pOztbffr0ka+vr4YMGaI777xTW7Zs0fz583XPPfeopKREP//5z9WzZ0998sknbdqeVatW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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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rXfHjh265ZZbdNttt7WrxRijHTt2/PWd7kBra6u++uora7l4OSFQb7/9ttLT0zVkyBC/+e68884OT3+PGTNGkZGR1npaWpokKScnxzp1fmn7sWPH/LZ3uVy6++67/dpmzpyptrY2vffee1ZN/fr105QpU/xqGjJkiNxud7uaBg8erNTU1Hb79t///d+6++67FR0drZCQEPXu3VsPPfSQWltbdeTIkcAO1BW69957/dYD3ReguxAA8K2TlpamrKwsZWVlaeLEifrVr36lCRMmaNGiRTp16pQkqb6+Xm632+8NSZLi4uIUGhqq+vp6v/ZZs2bp/PnzCg0N1bx58zp83dDQUEVHR/u1ud1u6/Uup76+Xh6Pp117QkLCX932m9x0003q3bu3tTz99NOdmufzzz/X/v37/ebq3bu3XC6XjDH64osv/Mb379/fbz0sLOwb28+ePevXHh8f366Grx/Hzz//XKdOnbLu7bh0qa2tbVdTR8e3urpaf//3f6/PPvtML7zwgv7whz+ovLxcL774oqQLNwp2t4iICPXt29evLdB9AboLTwHAFgYPHqzS0lIdOXJEt912m6Kjo/X+++/LGOMXAurq6vTVV18pJibGajtz5oz+8R//Uampqfr888/1ve99T2+88Ua71/jqq69UX1/vFwJqa2slqV0wuFR0dLROnDjRrv348eOS5FdLIN566y3rhkbpL4EiUDExMQoPD9fLL7982f7u9Pnnn7dr+/pxjImJUXR0tLZs2dLhHC6Xy2/960FPkjZt2qQzZ85o48aN8nq9Vvuf/vSnK661T58+kuR3nCVd9k27ozoC3ReguxAAYAsX/1G/eCf9uHHj9Jvf/EabNm3S1KlTrXElJSVW/0WPPvqoqqur9cEHH+jw4cOaPn26fvGLX2jBggXtXmft2rV+ZwjWrVsn6cKd5Jczbtw4FRUVad++fRo2bJhfLQ6HQ2PGjJEkOZ1OSVf+yTQjI+OKxl3kdDo7nHvy5MkqLCxUdHS0kpOTA5qzM06fPq0333zT7zLAunXr1KtXL91+++1WTevXr1dra6uGDx/eqde5+GZ88bhKkjFGq1evbjf2csfm4pMS+/fv15133mm1v/nmm1dcR3fsC9AZBAB863z44Yf66quvJF04Zbxx40Zt27ZNU6dOtd7AHnroIb344ovKy8tTVVWVMjIy9Mc//lGF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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAEnCAYAAADfOfvpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIgRJREFUeJzt3X1UlHX+//HXoDKCIggKA0nAlq50p3mfVkIqia6bmlub6WKWJ+/qmFuWeVrRXDnZsbWj5bZnN28yKts1texG1NTKLKOszST1CGabo4YiciOGfn5/+OP6OoLIwNCA1/Nxzhy7Ptdnrus9l9W85nN9rutyGGOMAACArQT4uwAAAPDrIwAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgDgsrR06VI5HA6PV9u2bZWUlKR33nnH3+VZ4uPjNWbMGK/fV1JSovT0dG3evNnnNdVWXl6eHA6Hli5d6tPtOhwOpaen+3SbDdHmzZvlcDga1N8pLm9N/V0AUJ+WLFmijh07yhgjt9utRYsWaciQIVq7dq2GDBni7/JqraSkRLNmzZIkJSUl+beY/y86OlqffvqprrrqKn+XAqAGCAC4rF133XXq1q2btTxw4EC1bt1ar732WqMOAA2R0+lUr169/F0GgBriFABspXnz5goMDFSzZs082o8dO6aJEyfqiiuuUGBgoH7zm99oxowZKisrkySdOnVKN954o66++mqdOHHCep/b7ZbL5VJSUpLOnDkjSRozZoxatmypXbt2qV+/fmrRooXatm2ryZMnq6Sk5JI1/vDDDxo1apQiIyPldDqVmJio+fPn6+zZs5LODbW3bdtWkjRr1izrFEdtTiVU5bHHHlNoaKj1eSTpoYceksPh0LPPPmu15efnKyAgQAsXLrTquvAUQHp6uhwOh3bt2qV77rlHoaGhioqK0tixYz2OoyQVFhZq3LhxioiIUMuWLTVw4EDt2bOnVp9h8eLF6tSpk1q2bKmQkBB17NhRTz75pLW+4hRRVlaW7rvvPoWHh6tFixYaMmSI9u/fX2l7GzZsUL9+/dSqVSsFBwerT58+2rhxY6V+e/fu1ciRIz3+7l544YVK/XJycjRw4EAFBwerTZs2Gj9+vE6ePFmrzwrUFgEAl7UzZ86ovLxcv/zyi3788UdNmTJFxcXFGjlypNXn1KlTSk5O1vLlyzV16lStW7dOo0aN0rx58zR8+HBJ54LDypUrdeTIEY0dO1aSdPbsWd17770yxui1115TkyZNrG3+8ssvGjRokPr166fVq1dr8uTJeumll3T33XdXW+/Ro0fVu3dvrV+/Xk8//bTWrl2r/v3769FHH9XkyZMlnRtqf//99yVJ999/vz799FN9+umneuqpp3xyzPr376/CwkJ9/vnnVtuGDRsUFBSkrKwsq23jxo0yxqh///6X3Oadd96pDh066D//+Y+eeOIJZWZm6pFHHrHWG2M0dOhQvfLKK/rzn/+st956S7169VJqaqrX9b/++uuaOHGi+vbtq7feekurV6/WI488ouLi4kp977//fgUEBCgzM1MLFizQ559/rqSkJBUUFFh9VqxYoZSUFLVq1UrLli3TypUrFR4erttvv90jBHz33Xfq3r27vv32W82fP1/vvPOOBg8erIcfftg6XSNJhw8fVt++ffXtt9/qxRdf1CuvvKKioiLr7xf41RjgMrRkyRIjqdLL6XSaF1980aPv3//+dyPJrFy50qP9mWeeMZLM+vXrrbY33njDSDILFiwwf/nLX0xAQIDHemOMSUtLM5LM888/79H+17/+1UgyH3/8sdUWFxdn0tLSrOUnnnjCSDKfffaZx3snTJhgHA6H+f77740xxhw9etRIMjNnzvT62FxKcXGxCQwMNLNnzzbGGPPjjz8aSebxxx83QUFB5tSpU8YYY8aNG2diYmKs9+Xm5hpJZsmSJVbbzJkzjSQzb948j31MnDjRNG/e3Jw9e9YYY8x7771X7THz5nNOnjzZhIWFVdun4t+PYcOGebR/8sknRpKZM2eOdSzCw8PNkCFDPPqdOXPGdOrUyfTo0cNqu/322027du3MiRMnKtXTvHlzc+zYMWOMMY8//rhxOBxm586dHv0GDBhgJJkPP/ywxp8VqAtGAHBZW758uXbs2KEdO3bovffeU1pamiZNmqRFixZZfTZt2qQWLVpoxIgRHu+tGFI//1feXXfdpQkTJuixxx7TnDlz9OSTT2rAgAFV7vvee+/1WK4Ydfjwww8vWu+mTZt0zTXXqEePHpVqMcZo06ZNl/7QVagYCal4VZxOqEpwcLBuuukmbdiwQZKUlZWlsLAwPfbYYzp9+rQ+/vhjSedGBWry61+Sfv/733ss33DDDTp16pSOHDki6f+OycWOmTd69OihgoIC3XPPPVqzZo1+/vnni/a9cH+9e/dWXFycVc+2bdt07NgxpaWlVTp+AwcO1I4dO1RcXKxTp05p48aNGjZsmIKDgz36Dho0SKdOndL27dutz3rttdeqU6dOdf6sQF0QAHBZS0xMVLdu3dStWzcNHDhQL730klJSUjRt2jRrmDc/P18ul0sOh8PjvZGRkWratKny8/M92seOHatffvlFTZs21cMPP1zlfps2baqIiAiPNpfLZe3vYvLz8xUdHV2pPSYm5pLvrc5VV12lZs2aWa/Zs2dX279///7avn27iouLtWHDBt12222KiIhQ165dtWHDBuXm5io3N7fGAeDCY+F0OiVJpaWlks59ruqOmTdGjx6tl19+WQcOHNCdd96pyMhI9ezZ0+P0RXXbd7lc1nE+fPiwJGnEiBEex69Zs2Z65plnZIzRsWPHlJ+fr/Lyci1cuLBSv0GDBkmSFUQq/n2rSS1AfeIqANjODTfcoA8++EB79uxRjx49FBERoc8++0zGGI8QcOTIEZWXl6tNmzZWW3FxsUaPHq0OHTro8OHDeuCBB7RmzZpK+ygvL1d+fr7HF5rb7ZZU+cvwfBERETp06FCl9p9++kmSPGrxxttvv21NaJT+L1BcTL9+/fTUU09p69at2rhxo2bOnGm1r1+/XgkJCdayL0RERFR7zLx133336b777lNxcbG2bt2qmTNn6ne/+5327NmjuLi4arfvdrt19dVXS/q/471w4cKLXuEQFRWl8vJyNWnSRKNHj9akSZOq7FdxzCIiIi66X+DXxAgAbGfnzp2SZM2k79evn4qKirR69WqPfsuXL7fWVxg/frx++OEHrVq1Sv/617+0du1a/e1vf6tyP6+++qrHcmZmpqTqr9vv16+fvvvuO3355ZeVanE4HEpOTpZU+Rf0pVx//fXWSEi3bt0uGQB69OihVq1aacGCBXK73dZpjv79++urr77SypUrdc0111xyOzVV8bkudsxqq0WLFkpNTdWMGTN0+vRp7dq1y2P9hfvbtm2bDhw4YP0d9enTR2FhYfruu+88jt/5r8DAQAUHBys5OVlfffWVbrjhhir7VQSb5ORk7dq1S19//bVPPyvgLUYAcFn79ttvVV5eLunc0OuqVauUlZWlYcOGWb/I/vSnP+mFF15QWlqa8vLydP31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BQAA6hEXFAMAYEMEAAAAbIgAAACADREAAACwIQIAAAA2RAAAAMCGCAAAANgQAQAAABsiAAAAYEMEAAAAbIgAAACADREAAACwIQIAAAA2RAAAAMCGCAAAANgQAQAAABsiAAAAYEMEAAAAbIgAAACADREAAACwoab+LgDApZWUlEiSvvzySz9XIpWWliovL0/x8fEKCgryay27d+/26/6BxowAADQCOTk5kqRx48b5uZKGKSQkxN8lAI0OAQBoBIYOHSpJ6tixo4KDg/1ay+7duzVq1CitWLFCiYmJfq1FOvfl3759e3+XATQ6BACgEWjTpo0eeOABf5fhITExUV26dPF3GQBqiUmAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCGfB4D09HQ5HA6Pl8vl8vVuAABAHdTLfQCuvfZabdiwwVpu0qRJfewGAADUUr0EgKZNm/KrHwCABqxe5gDs3btXMTExSkhI0B//+Eft37+/PnYDAABqyecjAD179tTy5cvVoUMHHT58WHPmzFHv3r21a9cuRUREVOpfVlamsrIya7mwsNDXJQEAgAv4fAQgNTVVd955p66//nr1799f69atkyQtW7asyv4ZGRkKDQ21XrGxsb4uCQAAXKDeLwNs0aKFrr/+eu3du7fK9dOnT9eJEyes18GDB+u7JAAAbK/enwZYVlam3bt365ZbbqlyvdPplNPprO8yAADAeXw+AvDoo49qy5Ytys3N1WeffaYRI0aosLBQaWlpvt4VAACoJZ+PAPz444+655579PPPP6tt27bq1auXtm/frri4OF/vCgAA1JLPA8Drr7/u600CAAAf41kAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYUFN/FwDg11NSUqKcnJw6bWP37t0ef9ZFx44dFRwcXOftAPAeAQCwkZycHHXt2tUn2xo1alSdt5Gdna0uXbr4oBoA3iIAADbSsWNHZWdn12kbpaWlysvLU3x8vIKCgupcDwD/cBhjjL+LOF9hYaFCQ0N14sQJtWrVyt/lAADQaHjzHcokQAAAbKjeAsCLL76ohIQENW/eXF27dtVHH31UX7sCAABeqpcA8MYbb2jKlCmaMWOGvvrqK91yyy1KTU3VDz/8UB+7AwAAXqqXOQA9e/ZUly5dtHjxYqstMTFRQ4cOVUZGRrXvZQ4AAAC149c5AKdPn1Z2drZSUlI82lNSUrRt2zZf7w4AANSCzy8D/Pnnn3XmzBlFRUV5tEdFRcntdlfqX1ZWprKyMmu5sLDQ1yUBAIAL1Nt9ABwOh8eyMaZSmyRlZGRo1qxZldoJAgAAeKfiu7MmZ/d9HgDatGmjJk2aVPq1f+TIkUqjApI0ffp0TZ061Vr+3//+p2uuuUaxsbG+Lg0AAFs4efKkQkNDq+3j8wAQGBiorl27KisrS8OGDbPas7KydMcdd1Tq73Q65XQ6reWWLVvq4MGDCgkJqXLEAIB/FRYWKjY2VgcPHmSiLtDAGGN08uRJxcTEXLJvvZwCmDp1qkaPHq1u3brppptu0j/+8Q/98MMPGj9+/CXfGxAQoHbt2tVHWQB8qFWrVgQAoAG61C//CvUSAO6++27l5+dr9uzZOnTokK677jq9++67iouLq4/dAQAALzW4ZwEAaNi4VwdweeBZAAC84nQ6NXPmTI+5OwAaH0YAAACwIUYAAACwIQIAAAA2RAAAAMCGCAAAANgQAQBAjWzdulVDhgxRTEyMHA6HVq9e7e+SANQBAQBAjRQXF6tTp05atGiRv0sB4AP19jRAAJeX1NRUpaam+rsMAD7CCAAAADZEAAAAwIYIAAAA2BABAAAAGyIAAABgQ1wFAKBGioqKtG/fPms5NzdXO3fuVHh4uK688ko/VgagNngaIIAa2bx5s5KTkyu1p6WlaenSpb9+QQDqhAAAAIANMQcAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA39P4PzWdwd8DFWAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAEnCAYAAADfOfvpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAJbFJREFUeJzt3Xt4VNWh/vF3kjAhQBJlgCRASAKiQoJCDZeDcoCiVBROI6KtXMrFWq5ViIpGKwJF8qBA9DlpKtJzQAsISgEpVJSqCJQIUYqCtF74cZUENFhygSQkWb8/OJlmkjC5MMnMuL+f59mP2Wuvnb0yPM5699pr720zxhgBAABLCfB2AwAAQNMjAAAAYEEEAAAALIgAAACABREAAACwIAIAAAAWRAAAAMCCCAAAAFgQAQAAAAsiAACVrFy5UjabzWVp27atBg0apC1btni7eU6xsbGaMGFCvfe7cOGC5s6dqx07dni8TfXRkPZX/NscO3bMWTZo0CAlJCTUaX+bzaa5c+c613fs2CGbzebyWcydO1c2m81lv4yMDK1cubJebQX8QZC3GwD4ohUrVujGG2+UMUY5OTlKT0/XiBEjtHnzZo0YMcLbzWuwCxcuaN68eZIud57esnHjRoWFhdVrn7vvvluZmZmKiopq0DEzMzPVsWNHt3V++ctf6s4773Qpy8jIUJs2bRoUuABfRgAAapCQkKDExETn+p133qlrr71Wr7/+ul8HAF/Rq1eveu/Ttm1btW3btsHH7NevX611OnbsWGtIAH4ouAQA1EHz5s1lt9vVrFkzl/Jz585p2rRp6tChg+x2uzp37qynn35axcXFkqSioiL16tVL1113nc6fP+/cLycnR5GRkRo0aJDKysokSRMmTFCrVq30+eefa8iQIWrZsqXatm2rGTNm6MKFC7W28cSJExo7dqzatWun4OBgdevWTUuWLFF5ebkk6dixY84OdN68ec5LHJ46s01KSlJMTIzzeJX17dtXP/rRj5zrVS8BlJeXa8GCBbrhhhsUEhKia665RjfddJNeeuklZ52aLgFU2LVrl/r166eQkBB16NBBzzzzjPNzrVD1EkBNql4CiI2N1eeff64PP/zQ+XnFxsaqoKBA11xzjSZPnlztdxw7dkyBgYF64YUX3B4L8DYCAFCDsrIylZaW6tKlSzp16pRmzpypwsJCjR492lmnqKhIgwcP1muvvabk5GRt3bpVY8eO1fPPP6+RI0dKuhwc3njjDZ09e1aTJk2SdLmzGzNmjIwxev311xUYGOj8nZcuXdJdd92lIUOGaNOmTZoxY4aWLVumn/3sZ27b++2336p///5699139dvf/labN2/W7bffrscee0wzZsyQJEVFRWnbtm2SpAcffFCZmZnKzMzUM88845HPbNKkSTpx4oTef/99l/J//vOf2rdvnyZOnHjFfZ9//nnNnTtXDzzwgLZu3ap169bpwQcf1L/+9a9aj5uTk6Of//znGjNmjN566y2NGjVKCxYs0COPPHK1f5I2btyozp07q1evXs7Pa+PGjWrVqpUmTZqk1atXuwQ76fIlA7vd7vz3BnyWAeC0YsUKI6naEhwcbDIyMlzqvvzyy0aSeeONN1zKFy1aZCSZd99911m2bt06I8m8+OKLZs6cOSYgIMBluzHGjB8/3kgyL730kkv5c889ZySZ3bt3O8tiYmLM+PHjnetPPvmkkWT27t3rsu/UqVONzWYzX3zxhTHGmG+//dZIMs8++2y9P5vaXLp0yURERJjRo0e7lM+ePdvY7Xbz3XffXbH9w4cPNz179nT7+yv+bY4ePeosGzhwoJFk3nrrLZe6Dz30kAkICDDHjx93llX9uz/44AMjyXzwwQfOsmeffdZU/VqMj483AwcOrNaeI0eOmICAAJOWluYsu3jxonE4HGbixIlu/xbAFzACANTgtddeU1ZWlrKysvT2229r/Pjxmj59utLT05113n//fbVs2VKjRo1y2bdiaPu9995zlt1///2aOnWqHn/8cS1YsEBPPfWU7rjjjhqPPWbMGJf1ilGHDz744Irtff/999W9e3f16dOnWluMMdXOyuuqYiSkYqlpeL9CUFCQxo4dqw0bNjjPisvKyvTHP/5RP/3pT+VwOK64b58+ffTpp59q2rRpeuedd5SXl1fnNoaGhuq//uu/XMpGjx6t8vJy7dy5s86/p746d+6s4cOHKyMjQ8YYSdKaNWuUm5vrHHUBfBkBAKhBt27dlJiYqMTERN15551atmyZhg4dqtmzZzuHpXNzcxUZGVnttrF27dopKChIubm5LuWTJk3SpUuXFBQUpIcffrjG4wYFBVXrKCMjI53Hu5Lc3NwaZ8e3b9++1n3d6dKli5o1a+Zc5s+f77b+pEmTVFRUpLVr10qS3nnnHWVnZ7sd/peklJQULV68WB999JGGDRsmh8OhIUOG6OOPP661jREREdXK6vKZecIjjzyir776Stu3b5ck/e53v9N//Md/uMx3AHwVAQCoo5tuukkXL17Ul19+KUlyOBw6c+aM8+yvwtmzZ1VaWqo2bdo4ywoLCzVu3Dhdf/31CgkJ0S9/+csaj1FaWlqt08rJyXEe70ocDoeys7OrlZ8+fVqSXNpSH3/+85+dIyFZWVn61a9+5bZ+xSjEihUrJF2+nbJ9+/YaOnSo2/2CgoKUnJys/fv369y5c3r99dd18uRJ/eQnP6l1AuSZM2eqldXlM/OEH//4x0pISFB6err27Nmj/fv3a/r06Y16TMBTCABAHR04cECSnDPphwwZooKCAm3atMml3muvvebcXmHKlCk6ceKENmzYoP/5n//R5s2blZaWVuNxVq9e7bK+Zs0aSe7v2x8yZIgOHz6s/fv3V2uLzWbT4MGDJUnBwcGSpIsXL7r5S/+tR48ezpGQxMRE54iCOxMnTtTevXu1e/du/fnPf9b48eNdJjrW5pprrtGoUaM0ffp0nTt3rsZZ/5Xl5+dr8+bNLmVr1qxRQECA/vM//7POx72S4OBgt5/Xww8/rK1btyolJUURERG67777rvqYQFPgOQBADQ4dOqTS0lJJl4eRN2zYoO3bt+uee+5RXFycJOkXv/iFfve732n8+PE6duyYevTood27d2vhwoW66667dPvtt0uS/vCHP2jVqlVasWKF4uPjFR8frxkzZuiJJ57Qrbfe6nLd3m63a8mSJSooKFDv3r21Z88eLViwQMOGDdNtt912xfbOmjVLr732mu6++27Nnz9fMTEx2rp1qzIyMjR16lRdf/31ki5fL4+JidFbb72lIUOGqHXr1mrTpo1iY2M99tk98MADSk5O1gMPPKDi4uI63WY4YsQI57MX2rZtq+PHj+vFF19UTEyMunbt6nZfh8OhqVOn6sSJE7r++uv1l7/8RcuXL9fUqVPVqVOnq/57evToobVr12rdunXq3Lmzmjdvrh49eji3jx07VikpKdq5c6d+85vfyG63X/UxgSbh5UmIgE+p6S6A8PBw07NnT7N06VJTVFTkUj83N9dMmTLFREVFmaCgIBMTE2NSUlKc9T777DMTEhLiMuPdGGOKiorMLbfcYmJjY833339vjLl8F0DL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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAEnCAYAAADfOfvpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAKgZJREFUeJzt3X1cFWX+//E3NwIHBBVUkEBwFQNDK7XyZg2MtCxdzXWrJYtubE1lW3O/meauYmvwTcusda3MXa0tb7Yy77Y1sYQsrEiz0tXK1rtWEcNEFASB6/eHP+brEUJujh1oXs/H4zxsrrnOzOdMwLzPzDUzHsYYIwAAYCue7i4AAAD8+AgAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAaJQlS5bIw8PD6dWuXTslJiZq3bp17i7PEh0drbvvvrve7ysuLlZaWpqysrJcXhN+PGlpafLw8HDpMqt+9vft2+fS5TZFd999t6Kjo91dBlyMAACXWLx4sbZs2aKcnBwtXLhQXl5eGjZsmNauXevu0hqluLhYM2fOJAA0c2PGjNGWLVvcXQbQpHi7uwD8NMTHx6t3797W9I033qg2bdpo2bJlGjZsmBsrQ10VFxfL39/f3WVcFBEREYqIiHB3GUCTwhEAXBR+fn7y8fFRixYtnNqPHTum8ePH65JLLpGPj49+9rOfadq0aSotLZUknT59WldeeaW6dOmiwsJC6315eXkKCwtTYmKiKioqJJ09LNmyZUvt3LlTSUlJCggIULt27ZSamqri4uIL1njgwAGNHj1a7du3l6+vr+Li4vTUU0+psrJSkrRv3z61a9dOkjRz5kzrFEdDTiX8kBUrVmjw4MHq0KGDHA6H4uLiNGXKFJ06dcrqM2/ePHl4eGjPnj3V3v/II4/Ix8dH3333ndW2ceNGJSUlKSgoSP7+/urfv7/eeecdp/dVHRLftm2bRo0apTZt2qhz586SpE8++US33367oqOj5XA4FB0drV//+tfav39/tfW///776tu3r/z8/HTJJZfoj3/8oxYtWlTjofEVK1aob9++CggIUMuWLXXDDTfo008/rfO2MsYoNDRUEyZMsNoqKirUpk0beXp66siRI1b73Llz5e3trePHjzt93nNFR0dr6NChWr9+vXr27CmHw6HY2Fj97W9/q7buDz/8UP3795efn5/Cw8M1depUnTlzps61Vzl69Kh+85vfKDIyUr6+vmrXrp369++vjRs3Wn0SExMVHx+vzZs3q0+fPnI4HNa2rfrZr1JWVqZZs2YpNjbWWt4999yjo0ePVlt3Xbf/kiVLdOmll1q/Ey+//HK9PyeaCQM0wuLFi40k8+GHH5ozZ86YsrIyc/DgQfPggw8aT09Ps379eqtvSUmJ6dGjhwkICDBPPvmk2bBhg/njH/9ovL29zU033WT1++qrr0xgYKAZOXKkMcaYiooKc91115n27dubQ4cOWf1SUlKMj4+P6dixo3n88cfNhg0bTFpamvH29jZDhw51qjMqKsqkpKRY0/n5+eaSSy4x7dq1M88//7xZv369SU1NNZLMuHHjjDHGnD592qxfv95IMvfdd5/ZsmWL2bJli9mzZ4/Ltt+f/vQn8/TTT5t//vOfJisryzz//POmU6dOZuDAgVafo0ePGh8fHzNt2jSn95aXl5vw8HBrOxljzN///nfj4eFhRowYYVauXGnWrl1rhg4dary8vMzGjRutfjNmzDCSTFRUlHnkkUdMZmamWbVqlTHGmNdee81Mnz7dvPnmmyY7O9ssX77cJCQkmHbt2pmjR49ay/jss8+Mn5+f6dGjh1m+fLlZs2aNuemmm0x0dLSRZPbu3Wv1ffzxx42Hh4e59957zbp168zKlStN3759TUBAgNm5c2edt9ftt99uunbtak1/+OGHRpJxOBzm1VdftdqHDBlirr766mqf91xRUVEmIiLCdOvWzbz88svm7bffNr/61a+MJJOdnW3127lzp/H39zfdunUzy5YtM6tXrzY33HCD6dixY7XPeSE33HCDadeunVm4cKHJysoyq1atMtOnTzfLly+3+iQkJJiQkBATHh5unn32WfP222+bBx980EgyEyZMsPpVVFSYG2+80QQEBJiZM2eazMxMs2jRInPJJZeYbt26meLiYqtvXbd/1e/z8OHDzdq1a80rr7xiunTpYiIjI01UVFSdPyeaBwIAGqXqD8b5L19fX7NgwQKnvs8//7yRZP7xj384tT/xxBNGktmwYYPVtmLFCiPJzJs3z0yfPt14eno6zTfmbACQZJ555hmn9scff9xIMu+//77Vdn4AmDJlipFkPvroI6f3jhs3znh4eJgvv/zSGHN25yvJzJgxo97bpr4qKyvNmTNnTHZ2tpFkPvvsM2veyJEjTUREhKmoqLDa3nrrLSPJrF271hhjzKlTp0xwcLAZNmyY03IrKirM5ZdfXuMOcfr06Resq7y83Jw8edIEBAQ4betf/epXJiAgwCkUVFRUmG7dujntGA8cOGC8vb3Nb3/7W6flFhUVmbCwMHPrrbfWYeuctWjRIiPJHDhwwBhjzKxZs0xsbKz5xS9+Ye655x5jjDFlZWUmICDAPProo9U+77mioqKMn5+f2b9/v9VWUlJigoODzdixY6222267zTgcDpOXl+e0TWJjY+sdAFq2bGkmTpxYa5+EhAQjyaxevdqp/f777zeenp5WvcuWLTOSzBtvvOHULzc310iyfv/quv0rKipMeHi46dmzp6msrLT67du3z7Ro0YIA8BPEKQC4xMsvv6zc3Fzl5ubqX//6l1JSUjRhwgTNnz/f6vPuu+8qICBAo0aNcnpv1SH1cw9T33rrrRo3bpwefvhhzZo1S48++qgGDRpU47rvuOMOp+nk5GRJ0qZNm36w3nfffVfdunXT1VdfXa0WY4zefffdC3/oGlRUVKi8vNx6VZ1O+CH/+c9/lJycrLCwMHl5ealFixZKSEiQJO3atcvqd8899+jbb791OlS8ePFihYWFaciQIZKknJwcHTt2TCkpKdVquPHGG5Wbm+t0akGSfvnLX1ar6eTJk3rkkUfUpUsXeXt7y9vbWy1bttSpU6ecasrOztZ1112ntm3bWm2enp669dZbnZb39ttvq7y8XHfddZdTXX5+fkpISKjXAMvrr79ekqztkJmZqUGDBun6669XZmamJGnLli06deqU1bc2V1xxhTp27GhN+/n5qWvXrk6nOzZt2qSkpCSFhoZabV5eXrrtttvqXHeVq6++WkuWLNGsWbP04Ycf/uBphMDAQP3iF79waktOTlZlZaXee+89SdK6devUunVrDRs2zGm7XnHFFQoLC7O2a123/5dffqlDhw4pOTnZ6XRJVFSU+vXrV+/PiqaPAACXiIuLU+/evdW7d2/deOONeuGFFzR48GBNnjzZOg9bUFCgsLCwaudi27dvL29vbxUUFDi133vvvTpz5oy8vb314IMP1rheb29vhYSEOLWFhYVZ6/shBQUF6tChQ7X28PDwC763Np07d1aLFi2s12OPPfaDfU+ePKkBAwboo48+0qxZs5SVlaXc3FytXLlSklRSUmL1HTJkiDp06KDFixdLkr7//nutWbNGd911l7y8vCTJOgc+atQopxpatGihJ554QsYYHTt2zKmGmrZBcnKy5s+frzFjxujtt9/Wxx9/rNzcXLVr186ppoKCAqedYpXz26rquuqqq6rVtWLFCqfxCxcSFRWlzp07a+PGjSouLtaWLVusAPDtt9/qyy+/1MaNG+VwOOq00zr/Z0eSfH19q33Oqp+pc9XUdiErVqxQSkqKFi1apL59+yo4OFh33XWX8vLynPrVtF3P/7k+cuSIjh8/bo21OfeVl5dnbde6bv+q5brqs6Lp4yoAXDQ9evTQ22+/ra+++kpXX321QkJC9NFHH8kY4xQC8vPzVV5e7vRN8tSpU7rzzjvVtWtXHTlyRGPGjNHq1aurraO8vFwFBQVOf8ir/pjW9Me9SkhIiA4fPlyt/dChQ5LkVEt9rF271hrQKP1foKjJu+++q0OHDikrK8v61i/JCkzn8vLy0p133qlnn31Wx48f19KlS1VaWqp77rnH6lNV85///Gf16dOnxnWev2M5P4wVFhZq3bp1mjFjhqZMmWK1l5aWVgsPISEhTgPvqpy/M6uq6/XXX1dUVFSNddVHUlKSVq9erezsbFVWVioxMVGBgYEKDw9XZmamNm7cqAEDBsjX17fR65LOfs7zP5NU/XPWRdu2bTVv3jzNmzdPBw4c0Jo1azRlyhTl5+dr/fr1Vr/atmvVz3Xbtm0VEhLi9L5zBQYGWv2kC2//quW66rOi6SMA4KLZvn27JFkj6ZOSkvSPf/xDq1at0i233GL1qxplnJSUZLU98MADOnDggD7++GPt3r1bo0aN0tNPP62HHnqo2npeffVVpyMES5culXR2NPUPSUpKUkZGhrZt26aePXs61eLh4aGBAwdKkrUTOfcbYW26d+9ep37S/+18z99RvfDCCzX2v+eeezR79mwtW7ZMS5YsUd++fRUbG2vN79+/v1q3bq1///vfSk1NrXMd59dkjKlW06JFi6qNQE9ISNBbb72l7777ztrJVFZW6rXXXnPqd8MNN8jb21vffPNNjacc6uv666/XwoULNW/ePPXp08fa0SUlJenNN99Ubm6u0tPTG72eKgMHDtSaNWt05MgRK0BVVFRoxYoVjVpux44dlZqaqnfeeUcffPCB07yioiKtWbPG6TTA0qVL5enpqWuvvVaSNHToUC1fvlwVFRW65pprfnA9dd3+l156qTp06KBly5Zp0qRJ1s/n/v37lZOTU2uYRfNEAIBL7NixQ+Xl5ZLOHkpcuXKlMjMzdcstt6hTp06SpLvuukt/+ctflJKSon379ql79+56//33lZ6erptuusk6Z7to0SK98sorWrx4sS677DJddtllSk1N1SOPPKL+/fs7nbf38fHRU089pZMnT+qqq65STk6OZs2apSFDhujnP//5D9b70EMP6eWXX9bNN9+sxx57TFFRUfrnP/+pBQsWaNy4ceratauks9+ioqKitHr1aiUlJSk4OFht27Z1yV3R+vXrpzZt2uiBBx7QjBkz1KJFC7366qv67LPPauwfGxurvn37KiMjQwcPHtTChQud5rds2VJ//vOflZKSomPHjmnUqFFq3769jh49qs8++0xHjx7Vc889V2tNQUFBuvbaazVnzhzrc2ZnZ+uvf/2rWrdu7dR32rRpWrt2rZKSkjRt2jQ5HA49//zz1jgDT8+zZxijo6P12GOPadq0afrPf/5j3SPiyJEj+vjjjxUQEKCZM2fWebtdd9118vDw0IYNG5zed/311yslJcX6b1f5wx/+oDVr1ui6667T9OnT5e/vr7/85S/VxlNcSGFhoQYOHKjk5GTFxsYqMDBQubm5Wr9+vUaOHOnUNyQkROPGjdOBAwfUtWtXvfXWW3rxxRc1btw4a8zC7bffrldffVU33XSTfve73+nqq69WixYt9O2332rTpk0aPny4brnlljpvf09PT/3pT3/SmDFjdMstt+j+++/X8ePHlZaWximAnyr3jkFEc1fTVQCtWrUyV1xxhZk7d645ffq0U/+CggLzwAMPmA4dOhhvb28TFRVlpk6davX7/PPPjcPhcBqxb8zZS/J69eploqOjzffff2+MOXsVQEBAgPn8889NYmKicTgcJjg42IwbN86cPHnS6f3nXwVgjDH79+83ycnJJiQkxLRo0cJceumlZs6cOU4j7Y0xZuPGjebKK680vr6+RlK15TRGTk6O6du3r/H39zft2rUzY8aMMdu2bTOSzOLFi6v1X7hwoXXZW2FhYY3LzM7ONjfffLMJDg42LVq0MJdccom5+eabzWuvvWb1qRoVf+4I/irffvut+eUvf2natGljAgMDzY033mh27NhR4zbcvHmzueaaa4yvr68JCwszDz/8sHVVx/Hjx536rlq1ygwcONAEBQUZX19fExUVZUaNGuV0eWJdXXnllUaS+eCDD6y2//73v0aSCQkJcRrFfu7nPVdUVJS5+eabqy07ISHBJCQkOLV98MEHpk+fPk6fs+r/RV2vAjh9+rR54IEHTI8ePUxQUJBxOBzm0ksvNTNmzDCnTp1yWv9ll11msrKyTO/evY2vr6/p0KGDefTRR82ZM2eclnnmzBnz5JNPmssvv9z4+fmZli1bmtjYWDN27Fjz9ddfO/Wt6/ZftGiRiYmJMT4+PqZr167mb3/7m0lJSeEqgJ8gD2OMcUvyABrp7rvv1uuvv66TJ0+6uxScY/Dgwdq3b5+++uord5fSLCUmJuq7777Tjh073F0KfuI4BQCgwSZNmqQrr7xSkZGROnbsmF599VVlZmbqr3/9q7tLA3ABBAAADVZRUaHp06crLy9PHh4e6tatm/7+979r9OjR9VqOMabaIMPzeXl5ufyJfq5SWVl5wXs+eHvz5xZNC6cAALhdVlaWdeXFD1m8eLFLn8PgSmlpaRccyLh3714eqYsmhQAAwO2Kior05Zdf1tqnU6dOtd7bwZ0OHTpk3UPih/To0UM+Pj4/UkXAhREAAACwIW4FDACADTW5USmVlZU6dOiQAgMDm+yAHwAAmiJjjIqKihQeHm7djOuHNLkAcOjQIUVGRrq7DAAAmq2DBw8qIiKi1j5NLgBU3df74MGDCgoKcnM1AAA0HydOnFBkZKS1L61NkwsAVYf9g4KCCAAAADRAXU6hMwgQAAAbIgAAAGBDBAAAAGyoUQEgIyNDHh4emjhxotVmjFFaWprCw8PlcDiUmJionTt3NrZOAE1ARUWFsrKytGzZMmVlZV3w/v0Amq4GB4Dc3FwtXLhQPXr0cGqfPXu25s6dq/nz5ys3N1dhYWEaNGiQioqKGl0sAPdZuXKlunTpooEDByo5OVkDBw5Uly5dtHLlSneXBqABGhQATp48qTvuuEMvvvii2rRpY7UbYzRv3jxNmzZNI0eOVHx8vF566SUVFxdr6dKlLisawI9r5cqVGjVqlLp3764tW7aoqKhIW7ZsUffu3TVq1ChCANAMNSgATJgwQTfffLOuv/56p/a9e/cqLy9PgwcPttp8fX2VkJCgnJycxlUKwC0qKir0+9//XkOHDtWqVavUp08ftWzZUn369NGqVas0dOhQ/c///A+nA4Bmpt73AVi+fLm2bdum3NzcavPy8vIkSaGhoU7toaGh2r9/f43LKy0tVWlpqTV94sSJ+pYE4CLavHmz9u3bp2XLllW7tainp6emTp2qfv36afPmzUpMTHRPkQDqrV4B4ODBg/rd736nDRs2yM/P7wf7nX8DAmPMD96UICMj44LP0QbgPocPH5YkxcfHq6KiQps3b9bhw4fVoUMHDRgwQPHx8U79ADQP9QoAW7duVX5+vnr16mW1VVRU6L333tP8+fOt53nn5eWpQ4cOVp/8/PxqRwWqTJ06VZMmTbKmq25jCKBpqPpdnj9/vl544QXt27fPmhcdHa3f/OY3Tv0ANA/1GgOQlJSkL774Qtu3b7devXv31h133KHt27frZz/7mcLCwpSZmWm9p6ysTNnZ2erXr1+Ny/T19bVu+8vtf4GmZ8CAAWrXrp2mTp2q+Ph4p0GA8fHxevTRR9W+fXsNGDDA3aUCqId6HQEIDAy0DvdVCQgIUEhIiNU+ceJEpaenKyYmRjExMUpPT5e/v7+Sk5NdVzWAH9W5p/CMMdYLQPPl8jsBTp48WRMnTtT48ePVu3dv/fe//9WGDRvq9GQiAE3P5s2blZ+fr4yMDO3YsUP9+vVTUFCQ+vXrp507dyo9PV35+fnavHmzu0sFUA8eponF+BMnTqhVq1YqLCzkdADQBCxbtkzJyckqKiqSj4+PFixYoG+++UadO3fW+PHjVVpaqqCgIC1dulS//vWv3V0uYGv12Yc2uccBA2haahsE+MwzzzAIEGimeBgQgFoNGDBA7du3ZxAg8BNDAABwQeeeKWQQIPDTQAAAUKvNmzfr6NGjDAIEfmIIAABqVXWHv9TUVO3Zs0ebNm3S0qVLtWnTJn399ddKTU116gegeWAQIIBaVQ3u27Fjh/r06VPtfv87duxw6gegeeAIAIBaDRgwQNHR0UpPT1dlZaXTvMrKSmVkZKhTp04MAgSaGQIAgFp5eXnpqaee0rp16zRixAinqwBGjBihdevW6cknn5SXl5e7SwVQD5wCAHBBI0eO1Ouvv67f//73Ts/16NSpk15//XWNHDnSjdUBaAjuBAigzmp6HDDf/IGmoz77UE4BAKiziooKbd++XTk5Odq+fbsqKircXRKABuIUAIA6mTx5sp5++mmVl5dbbQ8//LAeeughzZ49242VAWgIjgAAuKDJkydrzpw5CgkJ0YsvvqjDhw/rxRdfVEhIiObMmaPJkye7u0QA9cQYAAC1KisrU0BAgEJCQvTtt9/K2/v/DhyWl5crIiJCBQUFOnXqlHx8fNxYKQDGAABwmQULFqi8vFyzZs1y2vlLkre3tx577DGVl5drwYIFbqoQQEMQAADU6ptvvpEkDR06tMb5Ve1V/QA0DwQAALXq3LmzJGndunU1zq9qr+oHoHlgDACAWjEGAGg+GAMAwGV8fHz00EMP6ciRI4qIiNDChQt16NAhLVy4UBERETpy5Igeeughdv5AM8N9AABcUNV1/k8//bTGjh1rtXt7e+vhhx/mPgBAM8QpAAB1VlZWpgULFuibb75R586dNX78eL75A01IffahBAAAAH4iGAMAAABqRQAAAMCGCAAAANgQAQAAABsiAACos5KSEqWmpuqGG25QamqqSkpK3F0SgAbiKgAAdTJixAitXr26Wvvw4cO1atWqH78gANVwFQAAl6ra+fv4+GjKlCnas2ePpkyZIh8fH61evVojRoxwd4kA6okjAABqVVJSIn9/f/n4+KioqMjpxj9lZWUKDAxUWVmZiouL5XA43FgpAI4AAHCZhx9+WJI0adKkanf98/Hx0cSJE536AWgeCAAAavX1119LksaMGaNjx46pe/fuCgkJUffu3XXs2DHdd999Tv0ANA88DAhArWJiYrRhwwb16NFDxcXFVvuxY8cUEhIif39/qx+A5oMjAABqNWfOHEmydv59+vTRO++8oz59+ji1V/UD0DwQAADU6txr/b29vXXttdcqIiJC1157rby9vWvsB6Dp4yoAALXq3r27duzYodatW+v48ePV5lf9vsbHx+uLL7748QsEYOEqAAAuc+jQIUnSG2+8oZ07d8rT8+yfDU9PT+3cuVMrVqxw6gegeWAQIIBahYeH69ixY0pKSnJqr6ys1GWXXebUD0DzwREAALXKzs52mg4KCtKzzz5b7fDi+f0ANG0EAAC1KiwsdJqOjY1V165dFRsbW2s/AE0bgwAB1MrhcOj06dMX7Ofn58eVAICbMQgQgMuUlpZKkp599lkVFBQoPj5ewcHBio+PV0FBgZ588kmnfgCah3oFgOeee049evRQUFCQgoKC1LdvX/3rX/+y5htjlJaWpvDwcDkcDiUmJmrnzp0uLxrAj8fX11eS9Ic//EHBwcH64osvVFBQoC+++ELBwcGaOXOmUz8AzUO9AkBERIT+93//V5988ok++eQTXXfddRo+fLi1k589e7bmzp2r+fPnKzc3V2FhYRo0aJCKioouSvEALr5///vfks4eWszPz3eal5+fb/1+V/UD0Dw0egxAcHCw5syZo3vvvVfh4eGaOHGiHnnkEUlnDwmGhobqiSee0NixY+u0PMYAAE2Pl5eXKisrJUmBgYGaMWOGZs6cae38PT09VVFR4c4SAah++9AGB4CKigq99tprSklJ0aeffio/Pz917txZ27Zt05VXXmn1Gz58uFq3bq2XXnqpxuWUlpY6nTs8ceKEIiMjCQDARVBcXKzdu3c36L1XXXWVFQLO5enpqdzc3AYtMzY21nqYEIDGq08AqPeNgL744gv17dtXp0+fVsuWLfXmm2+qW7duysnJkSSFhoY69Q8NDdX+/ft/cHkZGRnWOUQAF9fu3bvVq1cvly6zsrKywcvcunWrevbs6dJ6ANRNvQPApZdequ3bt+v48eN64403lJKS4nQDEA8PD6f+xphqbeeaOnWqJk2aZE1XHQEA4HqxsbHaunVro5axa9cujR49Wq+88ori4uIaXQ8A96h3APDx8VGXLl0kSb1791Zubq6eeeYZ67x/Xl6eOnToYPXPz8+vdlTgXL6+voweBn4k/v7+LvvGHRcXx7d3oBlr9H0AjDEqLS1Vp06dFBYWpszMTGteWVmZsrOz1a9fv8auBgAAuFC9jgA8+uijGjJkiCIjI1VUVKTly5crKytL69evl4eHhyZOnKj09HTFxMQoJiZG6enp8vf3V3Jy8sWqHwAANEC9AsCRI0d055136vDhw2rVqpV69Oih9evXa9CgQZKkyZMnq6SkROPHj9f333+va665Rhs2bFBgYOBFKR4AADQMzwIAUC/btm1Tr169GMEPNEE8CwAAANSKAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0RAAAAsCECAAAANkQAAADAhggAAADYEAEAAAAbIgAAAGBDBAAAAGyIAAAAgA0R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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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gpOTE7y8vBAQECDdNVHdjohaBoUQQui6iMYg91ngRNT0SktLYWxsDH19fTg5OSErK0ta5+rqil9++QWVlZUoKSmBSqXSYaVEBMg/h7KngYganUqlwujRo1FZWYns7Gw899xz6N+/P5577jncvn0blZWVGD16NAMDUQvDngYiajIeHh64fv16jeXu7u51zt1ARH889jQQkU7NmjUL169fh7W1Ndzd3eHo6Ah3d3dYW1vj+vXrmDVrlq5LJCItsaeBiBpdRUUFTExMYGhoiPLycqjVammdnp4elEolKioqUFxcDENDQx1WSkQAexqISIc2bNggDXS0trbGF198gezsbHzxxRewtrZGSUkJKisrsWHDBl2XSkRa4DwNRNTorly5AuDhNNG//PIL9PUf/q9mwoQJGD9+PBwcHPDbb79J7YioZdCqp2Hjxo3w8vKCubk5zM3N4e3tjSNHjkjrhRBYtGgRHB0doVKp4Ovri4sXL8ref3R0NBQKBYKCgrQpi4iameqHVQUGBqKqqgpr167F1KlTsXbtWlRVVeGll17SaEdELYNWPQ3t27fHxx9/DA8PDwDA119/jdGjR+Ps2bPo1q0bVqxYgdWrV2PLli3o1KkTPvroIwwbNgxXrlyp8aCax2VlZeF///d/4ePjU/+jIaJmwcHBAQCwe/dubN++HVVVVdK6v//971AqlRrtiKhl0KqnYdSoURg+fDg6deqETp06YdmyZTA1NcWpU6cghMDatWsxb948jBkzBt27d8fXX3+NkpIS7Nix44n7VavVeOONN7B48WI888wzDTogItK96pkeS0tLIYRASEgIzp49i5CQEAghUFpaqtGOiFqGeg+EVKvViI6ORnFxMby9vZGRkYGcnBz4+/tLbZRKJQYNGoSkpKQn7mvJkiWwsbHB22+/Xd9yiKgZmTBhgvRnhUKBrVu3omfPnti6dSvatGlTazsiav60Hgj5888/w9vbG2VlZTA1NcW+ffvQtWtXKRjY2dlptLezs9OYQvZx//d//4evvvoK586d06qO8vJylJeXS+8LCwu12p6Ims6XX34p/dna2hqDBg2CiYkJiouLkZCQgDt37kjtpk+frqMqiUhbWvc0dO7cGefOncOpU6cwadIkhIWF4T//+Y+0XqFQaLQXQtRYVq2oqAjjxo2TbsPSRmRkJCwsLKSXs7OztodCRE2kehbISZMm4d69e9i1axe2bNmCXbt24d69e5g0aZJGOyJqGbTuaTA0NJQGQvbp0wcpKSlYt24dZs+eDeDhaOhHBzfduXOnRu9DtevXryMzMxOjRo2SllUPmNLX18eVK1fg7u5e67bh4eGYOXOm9L6wsJDBgagJlJSU4PLly1ptUz1hk5WVFRITE7Fjxw5cvnwZnp6eCA4OxsGDB6V2aWlpWtfk6ekJY2NjrbcjooZp8IyQQ4YMgbOzM6KiouDo6IgZM2ZI08NWVFTA1tYWy5cvx8SJE2tsW1ZWVmP++fnz56OoqAjr1q1Dp06dZM8WxxkhiZpGWloaevfuresyNKSmpqJXr166LoOo1ZB7DtWqp2Hu3LkIDAyEs7MzioqKEB0djfj4eMTGxkKhUGD69OmIiIhAx44d0bFjR0RERMDY2BjBwcHSPkJDQ+Hk5ITIyEgYGRmhe/fuGp/Rtm1bAKixnIh0w9PTE6mpqVpvt27dOnzzzTewtLREUFAQNm/ejLfeegsxMTG4d+8eQkND8f7779e7JiL642kVGnJzcxESEoLs7GxYWFjAy8sLsbGxGDZsGICHD6gpLS3Fe++9h/v376Nfv344evSoxhwNN2/e1Bg9TUTNm7Gxcb3+Vf/111/Dzs4Oa9aswebNmwEAmzdvhr6+Pj744AOsWLGisUsloibGB1YRUZOqqKhAeHg4Vq9ejZkzZyIyMpIPqSJqZvjAKiJqFgwNDfHGG28AAN544w0GBqIWjKGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKShaGBiIiIZGFoICIiIlkYGoiIiEgWhgYiIiKSRavQsHHjRnh5ecHc3Bzm5ubw9vbGkSNHpPVCCCxatAiOjo5QqVTw9fXFxYsXn7jPL774Aj4+PmjXrh3atWuHoUOH4vTp0/U7GiIiImoyWoWG9u3b4+OPP8aZM2dw5swZDB48GKNHj5aCwYoVK7B69WqsX78eKSkpsLe3x7Bhw1BUVFTnPuPj4/H666/jxIkTSE5ORocOHeDv749ff/21YUdGREREjUohhBAN2YGlpSVWrlyJt956C46Ojpg+fTpmz54NACgvL4ednR2WL1+OiRMnytqfWq1Gu3btsH79eoSGhsquo7CwEBYWFigoKIC5uXm9joWImkZaWhp69+6N1NRU9OrVS9flENFj5J5D6z2mQa1WIzo6GsXFxfD29kZGRgZycnLg7+8vtVEqlRg0aBCSkpJk77ekpAS///47LC0t61saERERNQF9bTf4+eef4e3tjbKyMpiammLfvn3o2rWrFAzs7Ow02tvZ2SErK0v2/ufMmQMnJycMHTr0ie3Ky8tRXl4uvS8sLNTiKIiIiEhbWoeGzp0749y5c8jPz8eePXsQFhaGhIQEab1CodBoL4SosawuK1aswM6dOxEfHw8jI6Mnto2MjMTixYu1LZ+IiIjqSevLE4aGhvDw8ECfPn0QGRmJZ599FuvWrYO9vT0AICcnR6P9nTt3avQ+1GbVqlWIiIjA0aNH4eXl9dT24eHhKCgokF63bt3S9lCIiIhICw2ep0EIgfLycri5ucHe3h5xcXHSuoqKCiQkJKB///5P3MfKlSuxdOlSxMbGok+fPrI+V6lUSrd+Vr+IiIio6Wh1eWLu3LkIDAyEs7MzioqKEB0djfj4eMTGxkKhUGD69OmIiIhAx44d0bFjR0RERMDY2BjBwcHSPkJDQ+Hk5ITIyEgADy9JLFiwADt27ICrq6vUU2FqagpTU9NGPFQiIiJqCK1CQ25uLkJCQpCdnQ0LCwt4eXkhNjYWw4YNAwDMmjULpaWleO+993D//n3069cPR48ehZmZmbSPmzdvok2b/3ZwbNiwARUVFRg7dqzGZy1cuBCLFi1qwKERERFRY2rwPA3NBedpIGq+OE8DUfPW5PM0EBER0Z8LQwMRERHJwtBAREREsjA0EBERkSwMDURERCQLQwMRERHJwtBAREREsjA0EBERkSwMDURERCQLQwMRERHJwtBAREREsmj1wCoialnS09NRVFSk6zJw6dIljf/qmpmZGTp27KjrMohaHIYGolYqPT0dnTp10nUZGsaNG6frEiRXr15lcCDSEkMDUStV3cOwbds2dOnSRae1lJaWIjMzE66urlCpVDqt5dKlSxg3blyz6IEhamkYGohauS5dujSLx1EPGDBA1yUQUQNxICQRERHJwtBAREREsjA0EBERkSwMDURERCQLQwMRERHJwtBAREREsjA0EBERkSwMDURERCQLQwMRERHJwtBAREREsjA0EBERkSwMDURERCQLQwMRERHJwtBAREREsmgVGjZu3AgvLy+Ym5vD3Nwc3t7eOHLkiLReCIFFixbB0dERKpUKvr6+uHjx4lP3u2fPHnTt2hVKpRJdu3bFvn37tD8SIiIialJahYb27dvj448/xpkzZ3DmzBkMHjwYo0ePloLBihUrsHr1aqxfvx4pKSmwt7fHsGHDUFRUVOc+k5OT8de//hUhISE4f/48QkJC8Oqrr+LHH39s2JERERFRo1IIIURDdmBpaYmVK1firbfegqOjI6ZPn47Zs2cDAMrLy2FnZ4fly5dj4sSJtW7/17/+FYWFhRo9FgEBAWjXrh127twpu47CwkJYWFigoKAA5ubmDTkkolYhLS0NvXv3RmpqKnr16qXrcpoNfi9ENck9h9Z7TINarUZ0dDSKi4vh7e2NjIwM5OTkwN/fX2qjVCoxaNAgJCUl1bmf5ORkjW0A4KWXXnriNkRERPTH09d2g59//hne3t4oKyuDqakp9u3bh65du0oneTs7O432dnZ2yMrKqnN/OTk5tW6Tk5PzxDrKy8tRXl4uvS8sLNT2UIhaPXtTBVT5V4HbHPNcTZV/FfamCl2XQdQiaR0aOnfujHPnziE/Px979uxBWFgYEhISpPUKheZfRiFEjWWPq882kZGRWLx4sZbVE/25TOxtiC4/TAR+0HUlzUcXPPxeiEh7WocGQ0NDeHh4AAD69OmDlJQUrFu3ThrHkJOTAwcHB6n9nTt3avQkPMre3r5Gr8LTtgGA8PBwzJw5U3pfWFgIZ2dnbQ+HqFX7LLUCf/1wC7p4euq6lGbj0uXL+OyTYPxF14UQtUBah4bHCSFQXl4ONzc32NvbIy4uDj179gQAVFRUICEhAcuXL69ze29vb8TFxWHGjBnSsqNHj6J///5P/FylUgmlUtnQ8olatZwHAqVtOwGOz+m6lGajNKcKOQ8aNP6b6E9Lq9Awd+5cBAYGwtnZGUVFRYiOjkZ8fDxiY2OhUCgwffp0REREoGPHjujYsSMiIiJgbGyM4OBgaR+hoaFwcnJCZGQkAOD999/Hiy++iOXLl2P06NHYv38/jh07hpMnTzbukRIREVGDaBUacnNzERISguzsbFhYWMDLywuxsbEYNmwYAGDWrFkoLS3Fe++9h/v376Nfv344evQozMzMpH3cvHkTbdr8d1BW//79ER0djfnz52PBggVwd3fHt99+i379+jXSIRIREVFjaPA8Dc0F52kg0sT5CGrH74Wopiafp4GIiIj+XBgaiIiISBaGBiIiIpKFoYGIiIhkYWggIiIiWRgaiIiISBaGBiIiIpKFoYGIiIhkYWggIiIiWRr8wCoiap5KSkoAPJwBUddKS0uRmZkJV1dXqFQqndZy6dIlnX4+UUvG0EDUSl2+fBkA8M477+i4kubp0WfiEJE8DA1ErVRQUBAAwNPTE8bGxjqt5dKlSxg3bhy2bduGLl266LQW4GFg6Nixo67LIGpxGBqIWilra2tMmDBB12Vo6NKlCx8SRdSCcSAkERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQki1ahITIyEn379oWZmRlsbW0RFBSEK1euaLTJzc3F+PHj4ejoCGNjYwQEBCA9Pf2p+167di06d+4MlUoFZ2dnzJgxA2VlZdodDRERETUZrUJDQkICJk+ejFOnTiEuLg6VlZXw9/dHcXExAEAIgaCgINy4cQP79+/H2bNn4eLigqFDh0ptarN9+3bMmTMHCxcuxKVLl/DVV1/h22+/RXh4eMOOjoiIiBqNvjaNY2NjNd5HRUXB1tYWqampePHFF5Geno5Tp07hwoUL6NatGwBgw4YNsLW1xc6dOzFhwoRa95ucnIwBAwYgODgYAODq6orXX38dp0+frs8xERERURNo0JiGgoICAIClpSUAoLy8HABgZGQktdHT04OhoSFOnjxZ534GDhyI1NRUKSTcuHEDhw8fxogRIxpSHhERETUirXoaHiWEwMyZMzFw4EB0794dAODp6QkXFxeEh4fjs88+g4mJCVavXo2cnBxkZ2fXua/XXnsNd+/excCBAyGEQGVlJSZNmoQ5c+bUuU15ebkUUgCgsLCwvodCREREMtS7p2HKlCn46aefsHPnTmmZgYEB9uzZg6tXr8LS0hLGxsaIj49HYGAg9PT06txXfHw8li1bhg0bNiAtLQ179+7FwYMHsXTp0jq3iYyMhIWFhfRydnau76EQERGRDAohhNB2o6lTpyImJgY//PAD3Nzcam1TUFCAiooK2NjYoF+/fujTpw/++c9/1trWx8cHL7zwAlauXCkt27ZtG/72t7/hwYMHaNOmZraprafB2dkZBQUFMDc31/aQiKgJpaWloXfv3khNTUWvXr10XQ4RPaawsBAWFhZPPYdqdXlCCIGpU6di3759iI+PrzMwAICFhQUAID09HWfOnHlir0FJSUmNYKCnpwchBOrKNEqlEkqlUpvyiYiIqAG0Cg2TJ0/Gjh07sH//fpiZmSEnJwfAw4CgUqkAALt27YKNjQ06dOiAn3/+Ge+//z6CgoLg7+8v7Sc0NBROTk6IjIwEAIwaNQqrV69Gz5490a9fP1y7dg0LFizAX/7ylyde1iAiIqI/jlahYePGjQAAX19fjeVRUVEYP348ACA7OxszZ85Ebm4uHBwcEBoaigULFmi0v3nzpkbPwvz586FQKDB//nz8+uuvsLGxwahRo7Bs2bJ6HBIRERE1hXqNaWiO5F6PIaI/Hsc0EDVvcs+hfPYEERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERycLQQERERLIwNBAREZEsDA1EREQkC0MDERERyaKv6wKIqHkrKSnB5cuX6729Wq1GTEwMAGDPnj1Qq9XQ09NrUE2enp4wNjZu0D6ISHsKIYTQdRGNobCwEBYWFigoKIC5ubmuyyFqNdLS0tC7d29dl6EhNTUVvXr10nUZRK2G3HMoexqI6Ik8PT2Rmpqq9XbHjx/HrFmzMHDgQPTp0wcPHjyAqakpzpw5g5MnT2LFihUYPHhwvWsioj8eexqIqNGp1Wp4eHjA2toav/32GzIzM6V1rq6usLa2Rl5eHtLT0xt8qYKIGk7uOVSrgZCRkZHo27cvzMzMYGtri6CgIFy5ckWjTW5uLsaPHw9HR0cYGxsjICAA6enpT913fn4+Jk+eDAcHBxgZGaFLly44fPiwNuURUTORmJiIzMxMpKamokePHkhOTkZRURGSk5PRo0cPpKamIiMjA4mJiboulYi0oFVoSEhI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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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dbUzVl5ebvHx8YQNBOA0Cs4YL7zwgtatW6d169bpr3/9q8aOHatbb71VCxYscGref/99RUdH67rrrgtYtv60xLGHka+//npNmDBB//Vf/6VZs2bp3nvv1dChQ0+47TFjxgRMZ2VlSZI++OCD7+z3/fff1/nnn6+LL764QS9mpvfff//7d/oE6urqdOTIEed17CmO73LBBRfowgsvDBjLyspSeXm51q9fH1S/6enpat26tZYvXy5Jys/P18CBAzV8+HCtXr1aBw8e1M6dO/X1119ryJAhkr49dbVixQqNHj1abdq0Cej/qquu0qFDh7RmzZqA7V577bVBvzcrV66U9O2f7bF+/etfB0w3pp9Ro0YFTPfu3VuSnNMbq1evVmlpqcaOHdvgz2f48OFat26dqqqqvncfKysrddddd6l79+6KiIhQRESE2rZtq6qqKn3xxRff+x6cah9VVVVat26drrnmGrVu3dpZPiYmhmug0MCP92op4Dg9e/ZscIHo9u3bdeedd+qGG27QWWedpf379ysxMbHBbXsJCQmKiIjQ/v37A8Z/85vfaOHChYqMjNTkyZNPuN2IiAi1a9cuYCwxMVGSGqzvWPv371dycnKD8aSkpO9d9mR++tOfBpy/nz59eoNrEY5X3++Jxur7ONV+W7durfT0dC1fvlwzZ87UihUrdOeddzrXunz88cfatWuXJDlhY//+/Tpy5Igef/xxPf744yfs8V//+lfA9Nlnn33SfTqR/fv3KyIiQvHx8QHjHTt2bFAXbD/Hfwa8Xq8kqbq6WpK0Z88eSWoQdI9VWlqq6OhoZ/pE+5iVlaUVK1bovvvuU9++fRUbGyuPx6OrrrrK2dbJnGofHo9HR48ePelnA6hH2MAZrXfv3vrb3/6mr776ShdffLHatWunTz75RGYWEDj27t2rI0eOqH379s5YVVWVsrOzde6552rPnj367W9/q7feeqvBNo4cOaL9+/cHfNmUlJRIavgFdKx27dqpuLi4wfju3bslKaCXYLzzzjvOxa7S/4eBk6nv90Rj9fsQTL+DBw/W/fffr7Vr1+qbb77R0KFDFRMTo759+yo/P1+7d+/Wueeeq86dO0uS4uLiFB4eruzsbN16660n7LFbt24B0435nYd27drpyJEjKi0tDQgcx+9/Y/r5PvXvz+OPP65LLrnkhDXHh57j99Hv9+vd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",
      "text/plain": [
       "<Figure size 600x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for col in df.columns:\n",
    "    plt.figure(figsize=(6,3))\n",
    "    plt.boxplot(df[col])\n",
    "    plt.title(f'Boxplot - {col}')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "a14484df-cdce-4a56-9890-a18f3d6ebcef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "distance_to_solar_noon outliers: 0\n",
      "temperature outliers: 0\n",
      "wind_speed outliers: 40\n",
      "sky_cover outliers: 0\n",
      "visibility outliers: 428\n",
      "humidity outliers: 83\n",
      "average_wind_speed outliers: 25\n",
      "average_pressure outliers: 31\n",
      "power_generated outliers: 107\n"
     ]
    }
   ],
   "source": [
    "for col in df.select_dtypes(include='number').columns:\n",
    "    Q1 = df[col].quantile(0.25)\n",
    "    Q3 = df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "\n",
    "    lower = Q1 - 1.5 * IQR\n",
    "    upper = Q3 + 1.5 * IQR\n",
    "\n",
    "    count = ((df[col] < lower) | (df[col] > upper)).sum()\n",
    "    print(col, \"outliers:\", count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "3f43afab-f6f5-4b70-9c97-89c63ba8c77c",
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in ['visibility','wind_speed','average_wind_speed','average_pressure', 'humidity','power_generated']:\n",
    "    Q1 = df[col].quantile(0.25)\n",
    "    Q3 = df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "\n",
    "    lower = Q1 - 1.5 * IQR\n",
    "    upper = Q3 + 1.5 * IQR\n",
    "\n",
    "    df[col] = df[col].clip(lower, upper)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "7e3e5efe-dad7-4f09-b536-41cb1e35f86b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "distance_to_solar_noon outliers: 0\n",
      "temperature outliers: 0\n",
      "wind_speed outliers: 0\n",
      "sky_cover outliers: 0\n",
      "visibility outliers: 0\n",
      "humidity outliers: 0\n",
      "average_wind_speed outliers: 0\n",
      "average_pressure outliers: 0\n",
      "power_generated outliers: 0\n"
     ]
    }
   ],
   "source": [
    "for col in df.select_dtypes(include='number').columns:\n",
    "    Q1 = df[col].quantile(0.25)\n",
    "    Q3 = df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "\n",
    "    lower = Q1 - 1.5 * IQR\n",
    "    upper = Q3 + 1.5 * IQR\n",
    "\n",
    "    count = ((df[col] < lower) | (df[col] > upper)).sum()\n",
    "    print(col, \"outliers:\", count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "0003e7c1-d03b-408f-93d8-8add438eb9c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_to_solar_noon  temperature  wind_speed  sky_cover  visibility  humidity  average_wind_speed  average_pressure  power_generated\n",
       "0.050401                70           10.6        0          10.0        36.5      13.0                29.95             31812.5            1\n",
       "0.668464                55           16.1        1          10.0        89.0      3.0                 30.15             0.0                1\n",
       "0.668552                62           11.2        4          10.0        78.0      10.0                29.94             0.0                1\n",
       "0.668571                62           16.1        0          10.0        78.0      17.0                30.05             0.0                1\n",
       "0.668821                57           19.3        3          10.0        83.0      24.0                29.85             0.0                1\n",
       "                                                                                                                                          ..\n",
       "0.352874                60           10.5        1          10.0        78.0      3.0                 29.90             8511.0             1\n",
       "0.353273                63           9.8         4          10.0        83.0      3.0                 29.92             0.0                1\n",
       "                        70           9.7         1          10.0        75.0      0.0                 29.91             8376.0             1\n",
       "                                     11.6        0          10.0        68.0      0.0                 29.87             8112.0             1\n",
       "1.141361                43           9.3         2          10.0        83.0      6.0                 30.03             0.0                1\n",
       "Name: count, Length: 2919, dtype: int64"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "fceb4dd9-0fd6-418c-a1d4-4b240dbc92b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_to_solar_noon    0\n",
       "temperature               0\n",
       "wind_speed                0\n",
       "sky_cover                 0\n",
       "visibility                0\n",
       "humidity                  0\n",
       "average_wind_speed        0\n",
       "average_pressure          0\n",
       "power_generated           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6cdacd2a-288d-4496-8339-02e19b55747e",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop(\"Power_generated\",axis=1)\n",
    "y=df[\"Power_generated\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a6fd2d4-a131-4c00-b074-20063ee3a24c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['sky_cover'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9782c802-c851-4333-b0c3-c212cb74f574",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "478c3719-b091-4612-9f40-59ce24018637",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'y_train' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[10]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[43my_train\u001b[49m\n",
      "\u001b[31mNameError\u001b[39m: name 'y_train' is not defined"
     ]
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c6dfdfa3-4816-4177-a4d5-aeb360288fc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0ce8c9e6-4e47-4317-a4dc-0e652f28b703",
   "metadata": {},
   "outputs": [],
   "source": [
    "models={\n",
    "    \"Linear Regression\":LinearRegression(),\n",
    "    \"Ridge Regreesion\":Ridge(alpha=1.0),\n",
    "    \"Lasso Regression\":Lasso(alpha=0.01),\n",
    "    \"Decision tree\":DecisionTreeRegressor(random_state=42),\n",
    "    \"random forest\":RandomForestRegressor(n_estimators=100,random_state=42),\n",
    "    \"KNN regressor\":KNeighborsRegressor(n_neighbors=5),\n",
    "    \"SVR regressor\":SVR(\n",
    "    kernel='rbf',\n",
    "    C=100,\n",
    "    epsilon=0.01,\n",
    "    gamma='scale')\n",
    "\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2188a110-7f82-40f1-ad39-b03da967067e",
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline={}\n",
    "for name, model in models.items():\n",
    "    pipeline[name] = Pipeline([\n",
    "        ('imputer', SimpleImputer(strategy='mean')),\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', model)\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8afa9c0b-7154-4449-b34b-2be9d5affa0f",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'x_train' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[14]\u001b[39m\u001b[32m, line 2\u001b[39m\n\u001b[32m      1\u001b[39m \u001b[38;5;28;01mfor\u001b[39;00m name, model \u001b[38;5;129;01min\u001b[39;00m pipeline.items():\n\u001b[32m----> \u001b[39m\u001b[32m2\u001b[39m     model.fit(\u001b[43mx_train\u001b[49m, y_train)\n\u001b[32m      4\u001b[39m     y_pred = model.predict(x_test)   \u001b[38;5;66;03m# single consistent name\u001b[39;00m\n\u001b[32m      6\u001b[39m     mae  = mean_absolute_error(y_test, y_pred)\n",
      "\u001b[31mNameError\u001b[39m: name 'x_train' is not defined"
     ]
    }
   ],
   "source": [
    "for name, model in pipeline.items():\n",
    "    model.fit(x_train, y_train)\n",
    "    \n",
    "    y_pred = model.predict(x_test)   # single consistent name\n",
    "    \n",
    "    mae  = mean_absolute_error(y_test, y_pred)\n",
    "    mse  = mean_squared_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    r2=r2_score(y_test,y_pred)\n",
    "    \n",
    "    print(name)\n",
    "    print(\"MAE :\", mae)\n",
    "    print(\"RMSE:\", rmse)\n",
    "    print(\"r2_score: \",r2)\n",
    "    print(\"-\" * 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ed57e82f-2912-4d79-b471-8b0005a96c03",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "pickle.dump(model, open('model.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4418078-3531-4ca8-842a-e2287cbd2168",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d694b374-b09c-43eb-b79e-49e70a5fd1a4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e3103b4-a1e9-4897-b2ed-2588e6397a9d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56ab00ff-53b1-47cc-b406-f6f8c147edf4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
