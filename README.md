# Universidad Peruana de Ciencias Aplicadas

## Trabajo Final

**Curso:** Procesamiento de Imagenes  
**Carrera:** Ciencias de la Computación  
**Sección:** SC61

---

### Integrantes:

| Código      | Nombres y Apellidos               |
|-------------|-----------------------------------|
| **U202210161**  | Diego Antonio Salinas Casaico     |
| **U202216148**  | Salvador Diaz Aguirre             |
| **U202215375**  | Ricardo Rafael Rivas Carrillo     |

---

**Año académico:** 2024

**Docente:** Luis Martin Canaval Sanchez

---
## Objetivo del Trabajo
 Este trabajo se basa en un modelo que analice tatuajes usando herramientas como Mask RCNN para identificar que figura está dibujada en el tatuaje, así poder identificar si pertenece a alguna banda criminal o no.

## Descripción del Dataset
El dataset **"tattoo_v0"** fue obtenido de la plataforma Hugging Face, específicamente del usuario **Drozdik**. Este conjunto de datos incluye un total de **4370 observaciones** con las siguientes variables:

- **image**: Imágenes de tatuajes generados por computadora sobre un fondo blanco.
- **text**: Descripciones cortas de los tatuajes.

### Variables Derivadas
Para enriquecer el análisis y mejorar la clasificación, se generaron las siguientes variables binarias basadas en la detección de características específicas en las imágenes:

- **skull**: 1 si la imagen contiene una calavera, 0 si no.
- **dragon**: 1 si la imagen contiene un dragón, 0 si no.
- **knife**: 1 si la imagen contiene un cuchillo, 0 si no.
- **star**: 1 si la imagen contiene una estrella, 0 si no.
- **demon**: 1 si la imagen contiene un demonio, 0 si no.
- **eye**: 1 si la imagen contiene un ojo, 0 si no.

### Entrenamiento del Modelo
El modelo fue entrenado utilizando un dataset adicional que combina imágenes de personas tatuadas y no tatuadas, proporcionando una base robusta para la clasificación.

## Conclusiones

## Licencia:

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1)
