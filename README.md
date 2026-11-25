# Reconocimiento facial

Sistema de verificación 1:1 e identificación 1:N con RetinaFace y TransFace-B.

## Uso local

Desde la raíz del repositorio, con Python 3.11 y Git:

```bash
make venv
make models
make enroll
make evaluate-images
```

`make enroll` inscribe la galería de `database/enrollment/`. `make evaluate-images` ejecuta el protocolo de referencia y guarda el informe en `outputs/evaluation/images.json`. `make test` ejecuta las pruebas automatizadas y `make help` lista los objetivos.

## Docker

```bash
make docker-build
make docker-evaluate-images
make docker-all
```

Los contenedores montan `database/`, `outputs/` y `models/` desde el anfitrión. `docker-all` inscribe las fotos y evalúa la galería local. `docker-shell` permite ejecutar los mismos comandos dentro de la imagen.

## Documentación

- [Sistema y API](docs/system.md)
- [Guía de uso y Docker](docs/operations.md)
- [Calibración, evaluación y resultados](docs/evaluation.md)
- [Informe de fortalezas y vulnerabilidades](docs/security.md)

## Autores

- Rafael Expósito Muñoz
- Jesús Martínez Ortiz
- Pablo Jiménez Castro
- Javier Martín Jurado
