# Guía de uso

Los comandos se ejecutan desde la raíz del repositorio. La [instalación local](../README.md#uso-local) prepara el entorno y los modelos.

## Inscripción y consultas

```bash
source .venv/bin/activate
reconocimiento-facial detect foto.jpg --output outputs/detection.jpg
reconocimiento-facial enroll persona foto.jpg
reconocimiento-facial users list
```

La [API de Python](system.md#api-de-python) permite consultar la galería con otra fotografía.

```python
from reconocimiento_facial import SistemaReconocimientoFacial

sistema = SistemaReconocimientoFacial()
identidad = sistema.identificar("consulta.jpg")
coincide = sistema.verificar("persona", "consulta.jpg")
```

La lectura depende de los formatos que admita OpenCV. Si OpenCV no decodifica un HEIC, se puede convertir con `ffmpeg -i foto.HEIC -frames:v 1 foto.png`.

`make enroll` inscribe las fotos de `database/enrollment/` y deduce las identidades de los nombres de archivo, omitiendo la extensión y el sufijo de año. Si hay varias detecciones, la inscripción por lotes utiliza el recorte alineado de la cara de mayor confianza.

## Catálogo y credenciales

El catálogo predeterminado es `database/registration/users.csv`. En la CLI se puede seleccionar otro con `--database ruta/users.csv` antes del subcomando. Make admite `DATABASE=ruta/users.csv` en los objetivos que trabajan con la galería local y `DATASET=ruta` para seleccionar las fotos de inscripción.

`users add ID` crea un registro pendiente de plantilla. `users remove ID` da de baja la identidad y conserva la copia de la foto para auditoría. La inscripción archiva el archivo recibido en `images/` junto al catálogo o guarda en PNG una entrada recibida como matriz. El catálogo almacena la ruta relativa de la copia.

AES-256-GCM cifra y autentica el catálogo y los registros. La clave maestra aleatoria se protege con una clave derivada de la contraseña mediante scrypt. HKDF-SHA-256 deriva las claves del catálogo y de cada identidad. Las imágenes archivadas se guardan en claro.

`BIOMETRIC_DB_PASSWORD` configura la contraseña de acceso. El valor de demostración es `biometria-demo`. La opción `--ask-password` solicita la contraseña actual y tiene prioridad sobre la variable de entorno. `UserDatabase(path, password=...)` admite una contraseña explícita desde Python.

```bash
reconocimiento-facial --ask-password database change-password
```

El comando crea un catálogo vacío si todavía no existe. En un catálogo existente, vuelve a proteger la clave maestra con la nueva contraseña. Las copias anteriores conservan la contraseña con la que se crearon. `database migrate` cifra un catálogo CSV en claro.

## Evaluación local

`make evaluate` utiliza 12 variaciones sintéticas por foto de inscripción. `make evaluate PROBES=database/mis_sondas` utiliza fotografías reales con la identidad indicada en el nombre. Los modos se pueden ejecutar también con `make evaluate-verification` o `make evaluate-identification`.

Los informes `verification.json` e `identification.json` se guardan en `outputs/evaluation/`. El primero recoge las tasas de falsas aceptaciones y falsos rechazos. El segundo recoge los aciertos de identificación, las falsas identificaciones y los aciertos acumulados por rango. La [evaluación de referencia](evaluation.md) usa su propia galería y sus propias fotos.

`make test` ejecuta las pruebas automatizadas e incluye la inscripción real cuando los modelos están disponibles.

## Docker

`make docker-build` prepara los modelos y construye la imagen para CPU. `make docker-models` prepara solo los modelos. `MODELS_DIR=/ruta/absoluta` selecciona el directorio de modelos para Docker.

`make docker-all`, `make docker-evaluate-images` y `make docker-shell` montan `database/` y `outputs/` del anfitrión bajo `/app`, y los modelos en modo de solo lectura. Las rutas personalizadas para `DATABASE`, `DATASET` y `PROBES` de `docker-all` deben estar disponibles dentro del contenedor, por ejemplo bajo `database/`.
