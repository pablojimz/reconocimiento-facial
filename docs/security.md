# Fortalezas y vulnerabilidades del reconocimiento facial

La evaluación de referencia utiliza una galería de 50 identidades, 50 fotos de control y 41 imágenes de estrés. Los umbrales de identificación y verificación son 0,406. El alcance y los resultados por imagen se recogen en el [protocolo de evaluación](evaluation.md) y su [informe de referencia](../outputs/evaluation/reference.json).

## Fortalezas observadas

- **Discriminación entre identidades.** El sistema identificó y verificó correctamente 45 de los 50 controles. Entre las capturas comparables de inscritos se registraron 0 falsas aceptaciones en 3 185 verificaciones cruzadas. En 14 capturas comparables de personas desconocidas de los grupos DIV, CIC y TBA hubo 0 falsas identificaciones y 0 falsas aceptaciones en 700 verificaciones
- **Control de la captura.** La API de imagen exige exactamente un rostro detectado. Rechazó las tres fotografías de grupo del conjunto de estrés
- **Protección del catálogo.** AES-256-GCM cifra y autentica las plantillas y el catálogo. Las pruebas de base de datos comprueban que se rechazan una contraseña incorrecta y datos cifrados manipulados

## Vulnerabilidades y límites observados

- **Suplantación por parecido facial.** Ashley Olsen y Oliver Phelps se aceptaron en ambos modos como sus respectivos gemelos inscritos
- **Aceptación de representaciones digitales.** Un modelo 3D y un retrato generado por IA se aceptaron como Keanu Reeves en ambos modos. El sistema compara fotografías, por lo que una coincidencia solo acredita similitud facial. Autenticar la presencia de una persona exige un control adicional de prueba de vida
- **Rechazo de inscritos.** La similitud quedó por debajo del umbral en 3 de 48 controles comparables y en 5 de 17 capturas comparables de estrés. Entre los casos figuran cambios de edad, oclusión, caracterización y baja calidad. Otros dos controles fallaron en la detección
- **Exposición de imágenes y credenciales de demostración.** Las copias de inscripción se guardan en claro fuera del catálogo. El acceso a esas imágenes depende de los permisos del directorio
