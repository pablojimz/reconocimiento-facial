# Sistema de reconocimiento facial del Grupo Maeb

El sistema inscribe, verifica e identifica personas a partir de fotografías. La galería de la práctica contiene 50 identidades e incluye a Maeb.

## Procesamiento y decisión

RetinaFace con ResNet-50 localiza cada cara y cinco puntos faciales: los ojos, la nariz y las comisuras de la boca. El detector aplica un umbral de confianza de 0,7 y una supresión de detecciones solapadas con umbral 0,4. La API de imagen exige una única detección para inscribir, identificar o verificar. Los cinco puntos permiten alinear la cara a la geometría de referencia de ArcFace en una imagen de 112 por 112 píxeles.

TransFace-B es un modelo _vision transformer_ (ViT) entrenado con Glint360K. Recibe la cara alineada en RGB, con los canales escalados al intervalo de -1 a 1, y genera un vector de 512 valores. La normalización L2 produce la plantilla facial. Se rechazan vectores con valores no finitos o norma cero. El instalador fija las revisiones del código de los modelos y comprueba el SHA-256 de los pesos.

[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface/tree/b984b4b775b2c4dced95c1eadd195a5c7d32a60b), [TransFace](https://github.com/DanJun6737/TransFace/tree/fb6cd56d04c7ea525e328219f5d5066adac03ada).

La similitud coseno mide la proximidad angular entre dos plantillas. La verificación 1:1 compara la consulta con la identidad reclamada y devuelve `True` si la similitud alcanza el umbral. La identificación 1:N busca la mayor similitud en la galería y devuelve el identificador si alcanza el umbral o `None` si no hay coincidencia. Los registros pendientes de plantilla quedan fuera de las comparaciones. Cada modo tiene su propio umbral configurable, con un valor predeterminado de 0,406. La [evaluación](evaluation.md) explica cómo se fijó ese valor.

## API de Python

`SistemaReconocimientoFacial` admite tres tipos de entrada. Cada fila de la tabla indica los métodos que usan esa entrada.

| Entrada                           | Inscripción                  | Verificación                 | Identificación             |
| --------------------------------- | ---------------------------- | ---------------------------- | -------------------------- |
| Fotografía como ruta o matriz BGR | `inscribir(id, foto)`        | `verificar(id, foto)`        | `identificar(foto)`        |
| Cara recortada como matriz BGR    | `inscribir_sd(id, cara)`     | `verificar_sd(id, cara)`     | `identificar_sd(cara)`     |
| Vector de 512 valores             | `inscribir_t(id, plantilla)` | `verificar_t(id, plantilla)` | `identificar_t(plantilla)` |

La entrada recortada se redimensiona a 112 por 112 cuando procede y pasa directamente al extractor. La entrada vectorial pasa al comparador o al catálogo. `verificar_t` e `identificar_t` lanzan `ValueError` ante una copia exacta de una plantilla inscrita.

La inscripción individual rechaza identificadores que ya existen en el catálogo. Si recibe una imagen, archiva una copia junto al catálogo y guarda su ruta relativa. La [guía de uso](operations.md) incluye ejemplos y describe el almacenamiento.
