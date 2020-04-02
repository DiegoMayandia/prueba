# Lab 1: Uso de SQL mediante Python

### IntroducciÃ³n
El objetivo del presente laboratorio es guiar al estudiante en el trabajo con bases de datos relacionales. Una base de datos relacional es un conjunto de datos organizado en tablas que se relacionan mediante llaves. Las bases de datos permiten acceder a los datos a travÃ©s de SQL (Structured Query Language).

En las siguientes secciones se guiarÃ¡ al estudiante a trabajar con bases de datos mediante programas escritos en el lenguaje de programaciÃ³n Python.

### Herramientas

En el presente laboratorio se utilizarÃ¡ la siguiente herramienta:

- PyCharm Community Edition 2018 (Entorno de Desarrollo Integrado para Python)


### Â¿CÃ³mo crearemos y utilizaremos la base de datos?

La base de datos la crearemos y utilizaremos desde nuestro programa en Python con la librerÃ­a sqlite3, que permite interactuar con el motor de base de datos SQLite. La estructura de nuestra base de datos serÃ¡ construida segÃºn el diagrama entidad relaciÃ³n mostrado acontinuaciÃ³n:

![Diagrama de Entidad RelaciÃ³n para LinioExp](images/diagrama_er.png)

### Â¡Manos a la Obra!

Primero debemos crear nuestro proyecto en PyCharm

#### ConfiguraciÃ³n de la base de datos

Luego, debemos crear el archivo **setup.py**. En este archivo crearemos la base de datos, crearemos las tablas y agregaremos datos de prueba a las tablas.

#### CreaciÃ³n de la base de datos

```python
import sqlite3

# ConexiÃ³n a la base de datos
con = sqlite3.connect("linioexp_parcial.db")

cursor = con.cursor()
```

#### CreaciÃ³n de las tablas
Una vez creada la base de datos "linioexp_parcial.db", procedemos a crear las tablas guiÃ¡ndonos del diagrama entidad relaciÃ³n mostrado anteriormente. Primero se debe crear aquella tabla cuyas llaves primarias serÃ¡n referenciadas en las demÃ¡s tablas. Para crear una tabla se utiliza la sentencia de SQL:

```sql
CREATE TABLE [nombre_tabla]
```

Ahora crearemos las tablas â€clienteâ€ y "pedido" con sus respectivos atributos utilizando las siguientes lÃ­neas de cÃ³digo:

```python
# CreaciÃ³n de tabla Cliente
# Consideraciones:
# ID numÃ©rico entero creciente y automÃ¡tico
# El texto no puede ser vacio
# La fecha de registro no puede estar vacia

cursor.execute("""
CREATE TABLE IF NOT EXISTS cliente(
    id_cliente INTEGER PRIMARY KEY AUTOINCREMENT,
    nombres TEXT NOT NULL,
    apellidos TEXT NOT NULL,
    documento_identidad TEXT NOT NULL,
    email TEXT NOT NULL,
    fecha_registro DATETIME
)""")

cursor.execute("""
CREATE TABLE pedido(
  id_pedido INT IDENTITY (1, 1),
  id_cliente INT,
  fecha_registro DATETIME NOT NULL,
  fecha_entrega DATETIME NULL,
  estado TEXT NOT NULL,
  precio_total DECIMAL(8,2) NOT NULL,
  cantidad_total INT NOT NULL,
  direccion TEXT NOT NULL,
  distrito TEXT NOT NULL,
  provincia TEXT NOT NULL,
  departamento TEXT NOT NULL,
  PRIMARY KEY (id_pedido),
  FOREIGN KEY (id_cliente) REFERENCES cliente(id_cliente)
)""")

# Confirmar la operacion en la base de datos
con.commit()
```

#### Ingreso de datos a las tablas

Las 4 tablas creadas en el apartado anterior se encuentran vacÃ­as, es decir,
no guardan ningÃºn dato. Para cargar datos a una tabla previamente creada,
utilizamos la sentencia:

```sql
INSERT INTO [nombre_tabla]
(atributo_1, atributo_2, atributo_3, atributo_4)
VALUES (_,_,_,_), (_,_,_,_), (_,_,_,_)
```

La sentencia permite agregar varios registros en la tabla a travÃ©s de
una sola ejecuciÃ³n. En primer lugar se indica los atributos los cuales
recibirÃ¡n un valor de la lista de grupos de valores indicado en la clÃ¡usula
\textbf{VALUES}. Es importante tener cuidado con el formato de las fechas.
En algunas ocasiones dependiendo de la versiÃ³n y/o configuraciÃ³n el formato
de la fecha puede variar. Es decir, podemos tener AÃ±o-mes-dÃ­a o dÃ­a-mes-aÃ±o.

#### Agregamos dos clientes

```python
# INSERTAR DATOS A LA TABLA CLIENTE
cursor.execute("""
INSERT INTO cliente(nombres, apellidos, documento_identidad, email, fecha_registro)
VALUES ('Alicia', 'Garcia Aguilar', '75230816', 'alicia.gaguilar@gmail.com', '05-12-2018'),
('Luis', 'Medina Delgago', '53707830', 'luis.medina.delgado@hotmail.com', '19-11-2018')
""")
con.commit()
```

#### Agregamos un pedido

```python
# IMPORTAMOS LIBRERIA PARA CREAR FECHAS
from datetime import datetime

# VALORES A INSERTAR
id_cliente = 1
fecha_registro = datetime.now()
fecha_entrega = None
estado = 'EN CAMINO'
precio_total = 35.5
cantidad_total = 5
direccion = 'Calle De Prueba 123'
distrito = 'San Pruebin'
provincia = 'Pruevincia'
departamento = 'Depruebamento'

# CONSULTA PARA RELLENAR VALORES
query = """
INSERT INTO pedido(id_cliente, fecha_registro, fecha_entrega, estado,
 precio_total, cantidad_total, direccion, distrito, provincia, departamento)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# INSERTAR DATOS A LA TABLA PEDIDO
cursor.execute(
    query,
    (id_cliente, fecha_registro, fecha_entrega, estado, precio_total, cantidad_total, direccion, distrito, provincia, departamento)
)

# GRABA LOS CAMBIOS
con.commit()
```

#### RecuperaciÃ³n de los datos insertados

Para elegir quÃ© datos queremos extraer de la base de datos y utilizarlos como variables desde Python utilizamos la siguiente sentencia SQL:

```sql
SELECT [nombre_atributo] FROM [nombre_tabla]}
```

Puedes colocar los nombres de los atributos que deseas o si deseas seleccionar todos los atributos de una tabla puedes utilizar el simbolo ' * ' .

```python
# SELECCIONA LAS COLUMNAS fecha_registro Y precio_total DE LA TABLA pedido
cursor.execute("SELECT fecha_registro, precio_total FROM pedido")
rows = cursor.fetchall()
# VISUALIZA CADA FILA DE LA TABLA
print("pedidos:")
for row in rows:
    print(row)

# SELECCIONA TODAS LAS COLUMNAS DE LA TABLA CLIENTE
cursor.execute("SELECT * FROM cliente")
rows = cursor.fetchall()
print("clientes:")
# VISUALIZA CADA FILA DE LA TABLA
for row in rows:
    print(row)
```

#### CreaciÃ³n de todas las tablas e ingreso de datos de prueba

1. Utiliza el archivo **crear_tablas.py** del Blackboard para crear las tablas que faltan.

2. Utiliza el archivo **insertar_datos.py** del Blackboard para insertar mÃ¡s datos en todas las tablas.

#### Consultas a la base de datos linioexp\_parcial

En esta secciÃ³n nos enfocaremos en atender las preguntas de los tomadores de decisiones de negocio de LinioExp. Por ello, utilizamos sentencias SQL para obtener los datos de las 4 tablas.

##### Pregunta 1: Â¿CuÃ¡ntos clientes utilizan LinioExp para comprar productos?

```sql
SELECT COUNT(*) AS "Num.Cliente"
FROM cliente
```

##### Pregunta 2: Obtener el cliente que realizÃ³ un pedido con destino al distrito del CALLAO.

```sql
SELECT c.*
FROM pedido p, cliente c
WHERE p.id_cliente = c.id_cliente
AND p.distrito = "CALLAO"
```

##### Pregunta 3: Â¿QuÃ© clientes realizaron compras el 24-12-2018?

```sql
SELECT c.nombres, c.apellidos, p.fecha_registro
FROM cliente c, pedido p
WHERE c.id_cliente = p.id_cliente
AND p.fecha_registro = "24-12-2018 01:15:20 pm"
```

### Â¡Pon a prueba lo aprendido!

##### Pregunta 4: Â¿QuiÃ©n o quiÃ©nes realizaron el mayor gasto comprando en LinioExp? (5 puntos - 8 minutos)

##### Pregunta 5: Â¿CuÃ¡l o cuÃ¡les son los producto mÃ¡s vendidos? (5 puntos - 8 minutos)

##### Pregunta 6: Â¿QuiÃ©n o quiÃ©nes realizaron la mayor cantidad de pedidos? (5 puntos - 8 minutos)

##### Pregunta 7: Â¿CuÃ¡ntos productos unicos existen? (5 puntos - 8 minutos)

### Siguientes Laboratorios
