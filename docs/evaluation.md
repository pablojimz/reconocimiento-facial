# Evaluación del reconocimiento facial

## Calibración de umbrales

La galería de inscripción contiene una fotografía por cada una de las 50 identidades. La calibración compara cada fotografía con 12 variaciones propias para formar 600 pares genuinos. Las variaciones incluyen cambios de geometría, resolución, iluminación, compresión y oclusión. Las comparaciones entre identidades distintas forman 30 625 pares impostores.

El máximo impostor fue 0,267 y el mínimo genuino, 0,544. El umbral 0,406 es el punto medio del intervalo óptimo de la malla de calibración. Para verificación se minimiza la media de las tasas de falsas aceptaciones y falsos rechazos. Para identificación se maximiza el acierto en conjunto abierto, que incluye el rechazo de consultas cuya identidad queda fuera de la galería. En las 200 particiones por identidad del protocolo sintético, la mitad de evaluación obtuvo un 100 % de acierto en la primera posición y ninguna falsa aceptación ni falso rechazo. Las variaciones comparten la fotografía de inscripción y tienden a ser más parecidas que una captura independiente.

```bash
make calibrate
make thresholds
```

Los archivos se escriben en `outputs/calibration/`. Las [plantillas de referencia](../database/evaluation/reference_templates.npz) permiten recalcular los umbrales con `.venv/bin/python -m scripts.optimize_thresholds --templates database/evaluation/reference_templates.npz`.

## Protocolo con fotografías reales y casos de estrés

El [manifiesto de pruebas](../database/evaluation/trials.json) define las imágenes, las identidades reclamadas y el resultado esperado para 50 controles y 41 casos de estrés. Cada control procede de una captura distinta de la utilizada para inscribir a la misma persona. El [registro de procedencia](../database/evaluation/control_sources.json) documenta fuentes, hashes y recortes.

El protocolo reconstruye en memoria la galería de 50 plantillas de referencia y usa un umbral de 0,406 en ambos modos. Cada consulta con una única cara detectada pasa por identificación y por 50 verificaciones. Las consultas con cero o varias detecciones se contabilizan como fallos de captura.

```bash
make evaluate-images
```

El resultado se guarda en `outputs/evaluation/images.json` con las decisiones, puntuaciones y datos del entorno. El [informe de referencia](../outputs/evaluation/reference.json) contiene el detalle por imagen. Los [vectores de consulta](../database/evaluation/reference_probes.npz) permiten recalcular las similitudes de referencia mediante el producto escalar de los vectores normalizados. Una nueva inferencia puede arrojar puntuaciones distintas según la plataforma y las dependencias.

## Resultados de referencia

| Medida | Controles | Casos de estrés |
|---|---:|---:|
| Imágenes que cumplen el resultado esperado | 45/50 | 32/41 |
| Identificaciones correctas de inscritos | 45/50 | 12/17 |
| Verificaciones correctas de la identidad propia | 45/50 | 12/17 |
| Falsos rechazos en capturas comparables de inscritos | 3/48 | 5/17 |
| Falsas aceptaciones en verificaciones cruzadas entre inscritos | 0/2352 | 0/833 |
| Fallos de captura | 2 | 6 |
| Errores técnicos | 0 | 0 |

El [informe de fortalezas y vulnerabilidades](security.md) interpreta estos resultados y recoge los casos de suplantación observados.
