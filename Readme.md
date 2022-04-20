# Proyecto Final Curso AED

- Mariano Fernández Abadía.
- Marzo/Abril 2022.

## Compilación

Los ejemplos se pueden compilar directamente (yo uso gcc):

```{bash}
g++ test.cpp -o out
./out
```

## Notas

- No he tenido tiempo de documentar lo que he hecho.
- Al menos he incorporado las siguientes características:
    1. Clase `GenericLayer` que sirve de base para las capas de la red. 
    2. He implementado las siguientes capas:
       - `WGLayer` que implementa una capa de peso+sesgo.
       - `ReLuLayer` que implementa un rectificador lineal.
       - `NormLayer` que implementa una capa de normalización.
       - `SoftMaxLayer` que implementa una capa _softmax_.
       - `ConvLayer` que implementa una capa que permite aplicar la función de convolución sobre entradas de 1 y 2 dimensiones.
       - `SigmoidLayer` que implementa la función sigmoide.
       - `LambdaLayer` permite utilizar una función definida por el usuario. La he añadido porque otorga flexibilidad.
    3. Las capas están construidas con plantillas para poder utilizar el tipo de dato más adecuado para la aplicación (float, double, ect.). Las capas solo funcionan con tipos en coma flotante. La idea que tenía inicialmente era hacer versiones en coma flotante y en enteros de las capas pero no he tenido tiempo de desarrollarlo.
    4. Parseo de datos de las capas en csv.
    5. Clase `Net` que construye y configura las capas con métodos más directos. Da una interfaz más cómoda para construir la red neuronal.
    6. Parseo de datos para construir una clase `Net` a partir de un archivo de configuración `.toml`. Utilizo este sistema en vez de json porque me parecía más directo como método de configuración.
    7. Manejo de errores. Mediante códigos a nivel de capa y manjeo de excepciones a nivel de `Net`.
    8. La capa de convolución tiene unos parámetros propios de configuración recogido en la estructura `ConvKernel` que define el kernel de la convolción.
- Lo más interesante en cuanto a rendimiento de la librería es que la entrada de una capa es la salida de la anterior eso nos permite ahorramos copiar datos.
- Los datos de entrada/salida están colocados en el heap. Esto es así porque al construir desde archivos de configuración las clases no puedo determinar el espacio que necesito en tiempo de compilación. Había pensado en crear mi propio allocator estático para poder reservar un espacio es la pila e ir colocando ahí los datos pero me ha parecido un poco 'sobreingeniar' para un ejercicio que no tiene un fin concreto. Tampoco he dispuesto de todo el tiempo que me habría gustado (estoy contento aún así).
- No me preocupa el heap framentation porque en teoría la memoria se reserva de forma secuencial y no se va a liberar en toda la ejecución.
- Lo último que puse es la gestión de errores. Creo que he probado todo pero si algo no compila borrad la línea (los ejemplos compilan todo).
- Cosas que añadiría con más tiempo: **allocador estático**, **soporte para introducir capas de convolución en el `.toml`** y **especificar las plantillas para tipos enteros con SFINAE** (esto último ya lo probé con alguna capa y de hecho se ha quedado el parser para enteros).
- Como nota adiccional he disfrutado bastante del proyecto y me ha sorprendido lo robusto que me ha salido el tema de la convolución (ha funcionado a la primera como esperaba y no me ha dado ningún fallo en la depuración).
