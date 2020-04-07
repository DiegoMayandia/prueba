# Bienvenidos al taller de Data Science!
- En este taller se desarrollarán temas claves para entender la finalidad y utilidad de Data Science. 
- Para ello, se revisará la metodología más utilizada (CRISP-DM) a partir de un caso práctico. 
- Al finalizar el taller, el asistente será capaz de aplicar diferentes herramientas de Data Science para poder dar transformar (ya sea en modelos predicitivos o visualizaciones) y dar sentido a los datos. 

## Introducción: Metodología CRISP-DM

- La metodología CRISP-DM (CRoss Industry Standard Process for Data Mining) muestra un enfoque estructurado para la planificación y ejecución de proyectos de Data Mining, Business Analytics y Data Science. 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/1280px-CRISP-DM_Process_Diagram.png" width="400" height="400">

- Según la encuesta publicada por KDnuggets (site que cubre los campos de Business Analytics, Data Mining y Data Science a través de entrevistas con líderes en el tema), la metodología CRISP-DM es la más utilizada en proyectos de Data Science).

<img src="https://78.media.tumblr.com/774ecde5f223970b03bedc3a68eb0fae/tumblr_inline_nv1porp78R1rmpjcz_500.png" width="500" height="500">

# Caso: ¿Quién sobrevive al Titanic?

## Business Understanding

- El hundimiento del Titanic es uno de los accidentes náuticos más conocidos de la historia. 
- Luego de impactar contra un iceberg, el Titanic se hundío y con ello 1502 personas de los 2224 pasajeros fallecieron. 
- Una de las principales causas de tan alarmentes cifras es que no exisitían botes salvavidas suficientes para los pasajeros y la tripulación.
- Si bien existe cierto elemento de fortuna en la supervivencia de los pasajeros, existen ciertos grupos de personas que tenían mayor probabilidad de salvarse (ej: mujeres, niñas, niños y la clase alta).
- En este caso, se busca determinar qué tipo de personas tenían mayor probabilidad de sobrevivir.

## Data understanding

Para poder trabajar con los datos y realizar las diferentes tareas propuestas, se deben importar ciertas librerías que dan las capacidades necesarias.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import requests
from ipywidgets import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle 
warnings.filterwarnings('ignore')
%matplotlib inline
```

    /Users/jdmorzan/anaconda2/envs/ipykernel_py3/lib/python3.6/site-packages/matplotlib/__init__.py:886: MatplotlibDeprecationWarning: 
    examples.directory is deprecated; in the future, examples will be found relative to the 'datapath' directory.
      "found relative to the 'datapath' directory.".format(key))


Leer los datos a partir de un archivo CSV(Comma Separated Values). Existen otros tipos de archivo.


```python
datos = pd.read_csv('train.csv', dtype={"Age": np.float64})
```

Mostrar el número de registros (filas) y variables (columnas) que tienen nuestros datos.


```python
print ("El archivo tiene {} registros y {} variables.".format(datos.shape[0], datos.shape[1]))
```

    El archivo tiene 891 registros y 12 variables.


Mostrar las 5 (por default) primeras filas de los datos. En caso queramos ver más filas se puede especificar.


```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Mostrar las 5 (por default) últimas filas de los datos. En caso queramos ver más filas se puede especificar.


```python
datos.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



Calcular estadísticos descriptivos de los datos. El .T es para fines de visualización.


```python
datos.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>891.0</td>
      <td>446.000000</td>
      <td>257.353842</td>
      <td>1.00</td>
      <td>223.5000</td>
      <td>446.0000</td>
      <td>668.5</td>
      <td>891.0000</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>891.0</td>
      <td>0.383838</td>
      <td>0.486592</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>891.0</td>
      <td>2.308642</td>
      <td>0.836071</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.0000</td>
      <td>3.0</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>714.0</td>
      <td>29.699118</td>
      <td>14.526497</td>
      <td>0.42</td>
      <td>20.1250</td>
      <td>28.0000</td>
      <td>38.0</td>
      <td>80.0000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>891.0</td>
      <td>0.523008</td>
      <td>1.102743</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>8.0000</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>891.0</td>
      <td>0.381594</td>
      <td>0.806057</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>6.0000</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>891.0</td>
      <td>32.204208</td>
      <td>49.693429</td>
      <td>0.00</td>
      <td>7.9104</td>
      <td>14.4542</td>
      <td>31.0</td>
      <td>512.3292</td>
    </tr>
  </tbody>
</table>
</div>



Mostrar el tipo de datos de las variables


```python
datos.dtypes.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PassengerId</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Survived</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Name</td>
      <td>object</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Age</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SibSp</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Parch</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ticket</td>
      <td>object</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fare</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cabin</td>
      <td>object</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Embarked</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



Mostrar las columnas con datos faltantes y el porcentaje de los mismos


```python
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns [mis_val_table_ren_columns['Missing Values'] != 0].sort_values('Missing Values', ascending=False)

missing_values_table(datos)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>77.104377</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>19.865320</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>0.224467</td>
    </tr>
  </tbody>
</table>
</div>



Graficar dos plots:
- Violinplot: Estimación de la función de densidad.
- Swarmplot: Muestra cómo se colocan los puntos en la función de densidad.


```python
sns.violinplot(x= "Age", data = datos.dropna())
sns.swarmplot(x="Age", data=datos.dropna(), color = "grey")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111f59da0>




![png](output_27_1.png)



```python
sns.violinplot(x= "Fare", data = datos.dropna())
sns.swarmplot(x="Fare", data=datos.dropna(), color = "grey")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x102595438>




![png](output_28_1.png)


Gráficos de barra para variables categóricas, tomando como criterio adicional una clase.


```python
sns.factorplot(x='Survived', col='Pclass', kind='count', data=datos);
```


![png](output_30_0.png)



```python
sns.factorplot(x='Survived', col='Sex', kind='count', data=datos);
```


![png](output_31_0.png)



```python
sns.factorplot(x='Survived', col='Embarked', kind='count', data=datos);
```


![png](output_32_0.png)


Graficar distribucion de datos numéricos.


```python
sns.distplot(datos.dropna().Fare, kde=False);
```


![png](output_34_0.png)



```python
sns.distplot(datos.dropna().Age, kde=False);
```


![png](output_35_0.png)


Mostrar gráficos de dispersión o de barra de variable contra variable.


```python
sns.pairplot(datos.dropna(), vars=["Pclass", "Age", "SibSp", "Parch", "Fare"], hue='Survived')
```




    <seaborn.axisgrid.PairGrid at 0x114a9d470>




![png](output_37_1.png)


Mostrar la correlación existente entre los datos.


```python
corr = datos.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.5, cmap = sns.diverging_palette(250, 10, as_cmap=True))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x114d490f0>




![png](output_39_1.png)


## Data preparation

Combinar dos variables para crear una nueva (Feature Engineering).


```python
datos["f_size"] = datos["SibSp"] + datos ["Parch"]
```

Eliminar variables que no sirven para predecir


```python
del datos["Cabin"] #NA
del datos["PassengerId"] #N
del datos["Name"] #N
del datos["SibSp"] #C
del datos["Parch"] #C
del datos["Ticket"] #N
```

Rellenar datos faltantes.


```python
datos["Age"] = datos["Age"].fillna(datos["Age"].mean())
```


```python
datos["Embarked"] = datos["Embarked"].fillna(datos["Embarked"].mode())
```

Eliminar datos faltantes


```python
datos = datos.dropna()
```

Mostrar las nuevas dimensiones de los datos.


```python
print ("El archivo tiene {first} registros y {second} variables.".format(first = datos.shape[0], second = datos.shape[1]))
```

    El archivo tiene 889 registros y 7 variables.


Eliminar posibles outliers


```python
# datos = datos[datos["Age"] < 100]
```


```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>f_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Convertir variables categóricas a variables discretas.


```python
le1 = preprocessing.LabelEncoder()
datos["Sex"] = le1.fit_transform(datos["Sex"])
le2 = preprocessing.LabelEncoder()
datos["Embarked"] = le2.fit_transform(datos["Embarked"])
```

Mostrar la relación entre los valores originales y codificados.


```python
pd.DataFrame({"Valor original":le1.classes_ , "Valor codificado": [i for i in range(len(le1.classes_))]})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Valor original</th>
      <th>Valor codificado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({"Valor original":le2.classes_ , "Valor codificado": [i for i in range(len(le2.classes_))]})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Valor original</th>
      <th>Valor codificado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>f_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Normalizar las variables.


```python
ascaler = MinMaxScaler()
ascaler.fit(np.asanyarray(datos["Age"]).reshape(-1, 1))
datos["Age"] = ascaler.transform(np.asanyarray(datos["Age"]).reshape(-1, 1))
fscaler = MinMaxScaler()
fscaler.fit(np.asanyarray(datos["Fare"]).reshape(-1, 1))
datos["Fare"] = fscaler.transform(np.asanyarray(datos["Fare"]).reshape(-1, 1))
```


```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>f_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.271174</td>
      <td>0.014151</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.472229</td>
      <td>0.139136</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.321438</td>
      <td>0.015469</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.434531</td>
      <td>0.103644</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.434531</td>
      <td>0.015713</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Exportar datos limpios


```python
datos.to_excel('data_clean.xlsx')
```

Mostrar (nuevamente) el gráfico de correlación.


```python
corr = datos.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.5, cmap = sns.diverging_palette(250, 10, as_cmap=True))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x115ff04e0>




![png](output_67_1.png)


## Modeling y Evaluation - ¿Cuál es mi problema? ¿Cuál es mi objetivo?

- Aprendizaje supervisado: existe una hipótesis a priori. Tengo clases que quiero predecir y un "profesor" que las enseña.
    - Clasificación: predecir a qué clase corresponde una observación).
    - Regresión: predecir un valor númerico.
- Aprendizaje no supervisado: no existe una hipótesis a priori. Exploratorio. 
    - Clustering: los datos se agrupan entre sí a partir de semejanzas y diferencias.

<img src="https://cdn-images-1.medium.com/max/1600/1*AZMDyaifxGVdwTV-1BN7kA.png" width="600" height="600">

Separar en vectores de características (feature vectors) y labels (clases).


```python
X = datos.drop("Survived", axis = 1).values
y = datos["Survived"].values
```

Separar datos en entrenamiento y testing


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Importar diferentes modelos predictivos


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
```

Definir función para graficar matriz de confusión


```python
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline

import seaborn as sns

def plot_confusion_matrix(cm, 
                          normalize=False,
                          title='Confusion matrix',
                          #cmap=sns.diverging_palette(220,10,as_cmap=True)):
                          cmap = 'Blues'):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2) #IMPORTANTE
    plt.xticks(tick_marks, [0,1], rotation=45) #IMPORTANTE
    plt.yticks(tick_marks , [0,1]) #IMPORTANTE

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```

#### Técnias supervisadas

Construir clasificador


```python
clf = RandomForestClassifier()
```

Entrenar clasificador


```python
clf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



Predecir con el modelo entrenado


```python
y_predicted = clf.predict(X_test)
```

Mostrar los resultados de clasificación (precision, recall, fmeasure)


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.79      0.85      0.82       101
               1       0.78      0.70      0.74        77
    
       micro avg       0.79      0.79      0.79       178
       macro avg       0.79      0.78      0.78       178
    weighted avg       0.79      0.79      0.78       178
    


Mostrar matriz de confusión.


```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[86 23]
     [15 54]]



![png](output_89_1.png)


Lo misma idea se sigue para los demás clasificadores. 


```python
clf = MLPClassifier(solver = "adam")
```


```python
clf.fit(X_train, y_train)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False)




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.87      0.86       106
               1       0.80      0.76      0.78        72
    
       micro avg       0.83      0.83      0.83       178
       macro avg       0.82      0.82      0.82       178
    weighted avg       0.83      0.83      0.83       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[92 17]
     [14 55]]



![png](output_95_1.png)



```python
clf = DecisionTreeClassifier()
```


```python
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.83      0.79        99
               1       0.75      0.66      0.70        79
    
       micro avg       0.75      0.75      0.75       178
       macro avg       0.75      0.74      0.75       178
    weighted avg       0.75      0.75      0.75       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[82 27]
     [17 52]]



![png](output_100_1.png)



```python
clf = LogisticRegression()
```


```python
clf.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.81      0.84      0.82       105
               1       0.75      0.71      0.73        73
    
       micro avg       0.79      0.79      0.79       178
       macro avg       0.78      0.78      0.78       178
    weighted avg       0.79      0.79      0.79       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[88 21]
     [17 52]]



![png](output_105_1.png)



```python
clf = KNeighborsClassifier()
```


```python
clf.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.81      0.84      0.82       105
               1       0.75      0.71      0.73        73
    
       micro avg       0.79      0.79      0.79       178
       macro avg       0.78      0.78      0.78       178
    weighted avg       0.79      0.79      0.79       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[88 21]
     [17 52]]



![png](output_110_1.png)



```python
clf = svm.SVC()
```


```python
clf.fit(X_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.83      0.84       111
               1       0.72      0.75      0.74        67
    
       micro avg       0.80      0.80      0.80       178
       macro avg       0.78      0.79      0.79       178
    weighted avg       0.80      0.80      0.80       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[92 17]
     [19 50]]



![png](output_115_1.png)



```python
clf = BernoulliNB()
```


```python
clf.fit(X_train, y_train)
```




    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.85      0.83       105
               1       0.77      0.73      0.75        73
    
       micro avg       0.80      0.80      0.80       178
       macro avg       0.79      0.79      0.79       178
    weighted avg       0.80      0.80      0.80       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[89 20]
     [16 53]]



![png](output_120_1.png)



```python
clf = GradientBoostingClassifier()
```


```python
clf.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  n_iter_no_change=None, presort='auto', random_state=None,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)




```python
y_predicted = clf.predict(X_test)
```


```python
print(classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.84      0.84       110
               1       0.74      0.75      0.74        68
    
       micro avg       0.80      0.80      0.80       178
       macro avg       0.79      0.79      0.79       178
    weighted avg       0.80      0.80      0.80       178
    



```python
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, normalize=False,
                      title='Matriz de confusion')
```

    Confusion matrix, without normalization
    [[92 17]
     [18 51]]



![png](output_125_1.png)



```python
clf.feature_importances_
```




    array([0.13, 0.44, 0.13, 0.17, 0.02, 0.1 ])



#### Probar técnica no supervisada

Importar técnica de clustering.


```python
from sklearn.cluster import KMeans
```

Inicializar técnica de clustering, entrenar el modelo y obtener las etiquetas.


```python
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels_
```

Asignar las etiquetas correspondientes a los datos.


```python
datos["clusters"] = labels
```


```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>f_size</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.271174</td>
      <td>0.014151</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.472229</td>
      <td>0.139136</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.321438</td>
      <td>0.015469</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.434531</td>
      <td>0.103644</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.434531</td>
      <td>0.015713</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Mostrar estadísticos por cluster.


```python
datos[datos["clusters"] == 0].describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>798.0</td>
      <td>0.387218</td>
      <td>0.487420</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>798.0</td>
      <td>2.284461</td>
      <td>0.841540</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>798.0</td>
      <td>0.674185</td>
      <td>0.468972</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>798.0</td>
      <td>0.381660</td>
      <td>0.153854</td>
      <td>0.0</td>
      <td>0.296306</td>
      <td>0.367921</td>
      <td>0.443956</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>798.0</td>
      <td>0.057061</td>
      <td>0.092138</td>
      <td>0.0</td>
      <td>0.015412</td>
      <td>0.025374</td>
      <td>0.051822</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>798.0</td>
      <td>1.511278</td>
      <td>0.805719</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>f_size</th>
      <td>798.0</td>
      <td>0.457393</td>
      <td>0.710251</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>clusters</th>
      <td>798.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos[datos["clusters"] == 1].describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>91.0</td>
      <td>0.340659</td>
      <td>0.476557</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>91.0</td>
      <td>2.549451</td>
      <td>0.734298</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>91.0</td>
      <td>0.428571</td>
      <td>0.497613</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>91.0</td>
      <td>0.241831</td>
      <td>0.186152</td>
      <td>0.004147</td>
      <td>0.063835</td>
      <td>0.233476</td>
      <td>0.367921</td>
      <td>0.798944</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>91.0</td>
      <td>0.111643</td>
      <td>0.122304</td>
      <td>0.015469</td>
      <td>0.053432</td>
      <td>0.061264</td>
      <td>0.091543</td>
      <td>0.513342</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>91.0</td>
      <td>1.747253</td>
      <td>0.625272</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>f_size</th>
      <td>91.0</td>
      <td>4.846154</td>
      <td>1.943211</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>clusters</th>
      <td>91.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Revertir transformaciones para mayor interpretabilidad.


```python
datos["Age"] = ascaler.inverse_transform(np.asanyarray(datos["Age"]).reshape(1, -1))[0]
datos["Fare"] = fscaler.inverse_transform(np.asanyarray(datos["Fare"]).reshape(1, -1))[0]
```


```python
datos["Sex"] = le1.inverse_transform(datos["Sex"])
datos["Embarked"] = le2.inverse_transform(datos["Embarked"])
```


```python
datos[datos["clusters"] == 0].describe(include = "all").T.fillna("-")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.387218</td>
      <td>0.48742</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2.28446</td>
      <td>0.84154</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>798.0</td>
      <td>2</td>
      <td>male</td>
      <td>538</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>30.7925</td>
      <td>12.2437</td>
      <td>0.42</td>
      <td>24</td>
      <td>29.6991</td>
      <td>35.75</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>29.2342</td>
      <td>47.2052</td>
      <td>0</td>
      <td>7.8958</td>
      <td>13</td>
      <td>26.55</td>
      <td>512.329</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>798.0</td>
      <td>3</td>
      <td>S</td>
      <td>567</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>f_size</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.457393</td>
      <td>0.710251</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>clusters</th>
      <td>798.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos[datos["clusters"] == 1].describe(include = "all").T.fillna("-")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.340659</td>
      <td>0.476557</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2.54945</td>
      <td>0.734298</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>91.0</td>
      <td>2</td>
      <td>female</td>
      <td>52</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>19.6649</td>
      <td>14.814</td>
      <td>0.75</td>
      <td>5.5</td>
      <td>19</td>
      <td>29.6991</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>57.1982</td>
      <td>62.6597</td>
      <td>7.925</td>
      <td>27.375</td>
      <td>31.3875</td>
      <td>46.9</td>
      <td>263</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>91.0</td>
      <td>3</td>
      <td>S</td>
      <td>77</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>f_size</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>4.84615</td>
      <td>1.94321</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>clusters</th>
      <td>91.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Mostrar los gráficos de dispersión y de barras de variable contra variable.


```python
sns.pairplot(datos.dropna(), vars=["Pclass", "Age", "f_size", "Fare"], hue='clusters')
```




    <seaborn.axisgrid.PairGrid at 0x116b6ccc0>




![png](output_144_1.png)


## Deployment


```python
datos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>f_size</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Definir variables para probar el clasificador.


```python
Pclass = 3
Sex = 0
Age = ascaler.transform(np.asanyarray(30).reshape(1,-1))[0][0]
Fare = fscaler.transform(np.asanyarray(60).reshape(1,-1))[0][0]
Embarked = 2
f_size = 2
prueba = [Pclass, Sex, Age, Fare, Embarked, f_size]
```

Mostrar la predicción.


```python
prediction = clf.predict(np.asanyarray(prueba).reshape(1, -1))[0]
if prediction == 0:
    print ("No sobrevivio :(")
else:
    print ("Sobrevivio!")
```

    No sobrevivio :(


Predecir para nuevos datos.


```python
datosxpred = pd.read_csv("test.csv")
```


```python
datosxpred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
datosxpred["f_size"] = datosxpred["SibSp"] + datosxpred["Parch"]
```


```python
del datosxpred["Cabin"] #NA
del datosxpred["PassengerId"]
del datosxpred["Name"]
del datosxpred["SibSp"]
del datosxpred["Parch"]
del datosxpred["Ticket"]
```


```python
datosxpred["Age"] = datosxpred["Age"].fillna(datosxpred["Age"].mean())
```


```python
datosxpred["Embarked"] = datosxpred["Embarked"].fillna(datosxpred["Embarked"].mode())
```


```python
datosxpred = datosxpred.dropna()
```


```python
le1 = preprocessing.LabelEncoder()
datosxpred["Sex"] = le1.fit_transform(datosxpred["Sex"])
le2 = preprocessing.LabelEncoder()
datosxpred["Embarked"] = le2.fit_transform(datosxpred["Embarked"])
```


```python
ascaler = MinMaxScaler()
ascaler.fit(np.asanyarray(datosxpred["Age"]).reshape(-1, 1))
datosxpred["Age"] = ascaler.transform(np.asanyarray(datosxpred["Age"]).reshape(-1, 1))
fscaler = MinMaxScaler()
fscaler.fit(np.asanyarray(datosxpred["Fare"]).reshape(-1, 1))
datosxpred["Fare"] = fscaler.transform(np.asanyarray(datosxpred["Fare"]).reshape(-1, 1))
```

Añadir nueva variable de predicción.


```python
probs = clf.predict_proba(datosxpred)
```


```python
probs = pd.DataFrame(probs, columns= ["Prob. No sobrevivir", "Prob. Sobrevivir"])
```


```python
datosxpred["Survived"] = clf.predict(datosxpred)
```


```python
datosxpred = pd.concat([datosxpred, probs], axis = 1)
```

Exportar datos.


```python
datosxpred.to_excel("data_predicted.xlsx")
```
