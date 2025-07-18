# Workshop Secure MLOps - BSides

## Descripción

Este workshop demuestra las características de seguridad esenciales para MLOps, incluyendo escaneo de seguridad de modelos, pruebas de robustez adversarial, métricas de seguridad y integración con MLflow, todo corriendo en un entorno Docker.

## Características de Seguridad Incluidas

- 🔍 **Escaneo de Seguridad de Modelos** con ModelScan
- ⚔️ **Pruebas de Robustez Adversarial** con ART (Adversarial Robustness Toolbox)
- 📊 **Seguimiento de Métricas de Seguridad**
- 🔗 **Integración con MLflow** con enfoque en seguridad
- 🛡️ **Evaluación Segura de Modelos**

## Prerrequisitos

- Docker Desktop instalado y ejecutándose
- Python 3.8 o superior
- Jupyter Notebook o JupyterLab
- Navegador web para acceder a MLflow UI

## Configuración del Entorno

### 1. Configuración de Docker Desktop

1. **Descargar e instalar Docker Desktop:**
   - Visita [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Descarga la versión para tu sistema operativo
   - Sigue las instrucciones de instalación

2. **Verificar la instalación:**
   ```bash
   docker --version
   docker-compose --version
   ```

### 2. Configurar MLflow con Docker

1. **Usar el archivo docker-compose.yml existente o crear uno nuevo:**
   ```bash
   docker-compose up -d
   ```

2. **Verificar que MLflow está corriendo:**
   - Abre tu navegador y ve a `http://localhost:5001`
   - Deberías ver la interfaz de MLflow

### 3. Configuración del Entorno Python

1. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

2. **Instalar dependencias base:**
   ```bash
   pip install --upgrade pip
   pip install jupyter notebook
   ```

### 4. Instalar Paquetes de Seguridad

Los siguientes paquetes se instalarán automáticamente al ejecutar el notebook:

```bash
pip install mlflow
pip install scikit-learn pandas numpy matplotlib seaborn
pip install adversarial-robustness-toolbox
pip install modelscan
pip install cryptography
```

## Ejecutar el Workshop

### 1. Iniciar Jupyter Notebook

```bash
jupyter notebook
```

### 2. Abrir el Notebook del Workshop

1. En la interfaz de Jupyter, navega hasta `secure_mlops_demo.ipynb`
2. Abre el notebook

### 3. Ejecutar las Celdas del Workshop

El notebook está organizado en las siguientes secciones:

1. **Setup y Dependencias** - Instala todos los paquetes necesarios
2. **Configuración de MLflow** - Conecta al contenedor Docker de MLflow
3. **Preparación de Datos** - Genera datos sintéticos con metadatos de seguridad
4. **Entrenamiento de Modelos** - Entrena modelos con seguimiento de seguridad
5. **Escaneo de Seguridad** - Escanea modelos con ModelScan
6. **Pruebas Adversariales** - Prueba robustez con ART
7. **Dashboard de Seguridad** - Visualiza métricas de seguridad
8. **Resumen del Workshop** - Resumen completo de características

### 4. Acceder a MLflow UI

Durante y después de ejecutar el notebook:

- Abre `http://localhost:5001` en tu navegador
- Explora los experimentos y métricas de seguridad
- Revisa los artefactos de modelos y reportes de seguridad

## Estructura de Archivos Generados

Después de ejecutar el workshop, se crearán los siguientes directorios:

```
secmlops/
├── models/                           # Modelos entrenados
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── svm_model.pkl
├── artifacts/                        # Reportes de análisis de seguridad
│   ├── random_forest_security_analysis_*.txt
│   ├── logistic_regression_security_analysis_*.txt
│   └── svm_security_analysis_*.txt
└── secure_mlops_demo.ipynb          # Notebook principal
```

## Características Demostradas

### Seguridad de Datos
- Seguimiento de linaje de datos
- Hash de integridad de datos
- Metadatos de privacidad

### Seguridad de Modelos
- Escaneo de vulnerabilidades con ModelScan
- Análisis de complejidad del modelo
- Detección de posible sobreajuste

### Robustez Adversarial
- Ataques FGSM (Fast Gradient Sign Method)
- Ataques PGD (Projected Gradient Descent)
- Métricas de robustez

### Integración MLflow
- Logging de métricas de seguridad
- Artefactos de modelos con metadatos
- Seguimiento de experimentos seguros

## Solución de Problemas

### MLflow no se conecta
```bash
# Verificar que Docker está corriendo
docker ps

# Reiniciar MLflow
docker-compose down
docker-compose up -d
```

### Errores de paquetes de Python
```bash
# Reinstalar paquetes problemáticos
pip install --force-reinstall modelscan
pip install --force-reinstall adversarial-robustness-toolbox
```

### Problemas de permisos en Docker
```bash
# En sistemas Unix, puede ser necesario usar sudo
sudo docker-compose up -d
```

## Próximos Pasos

Después de completar el workshop:

1. **Explorar MLflow UI** para análisis detallado de experimentos
2. **Implementar herramientas de seguridad adicionales** (Cosign, Garak)
3. **Configurar monitoreo continuo de seguridad**
4. **Integrar con pipelines CI/CD**
5. **Expandir a escenarios de producción**
6. **Implementar pruebas de seguridad automatizadas**

## Recursos Adicionales

- [Documentación de ModelScan](https://github.com/protectai/modelscan)
- [Documentación de ART](https://adversarial-robustness-toolbox.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Desktop Documentation](https://docs.docker.com/desktop/)

## Soporte

Si encuentras problemas durante el workshop:

1. Revisa los logs de Docker: `docker-compose logs`
2. Verifica que todos los puertos estén disponibles
3. Asegúrate de que Docker Desktop esté ejecutándose correctamente
4. Consulta la sección de solución de problemas arriba

---

**¡Felicidades!** 🎉 Ahora tienes un entorno completo de Secure MLOps ejecutándose con Docker Desktop.