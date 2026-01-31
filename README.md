# Signa
Reconocimiento de Lengua de Signos mediante Deep Learning

Este proyecto desarrolla un sistema de reconocimiento automático de palabras en Lengua de Signos utilizando Deep Learning, con el objetivo de facilitar la comunicación entre personas sordomudas y oyentes en contextos sanitarios de urgencia.

El modelo se entrena a partir de secuencias temporales de puntos clave del cuerpo, manos y rostro, extraídos mediante técnicas de pose estimation usando HandLandmarker de MediaPipe. A partir de estos datos, se construye una arquitectura basada en redes neuronales profundas para clasificación secuencial, capaz de identificar palabras concretas representadas mediante gestos.

Características principales

Preprocesamiento de secuencias temporales de gestos.

Modelado secuencial con redes neuronales profundas.

Clasificación multiclase de palabras en Lengua de Signos.

Evaluación avanzada mediante Top-K Accuracy y matrices de confusión.

Estrategias de mejora del rendimiento como:

Ajuste de hiperparámetros.

Técnicas de regularización y funciones de pérdida avanzadas.

Objetivo del proyecto

Demostrar la viabilidad de aplicar técnicas modernas de Deep Learning para el reconocimiento de Lengua de Signos, donde una cámara pueda detectar gestos en tiempo real y traducirlos automáticamente a palabras.
