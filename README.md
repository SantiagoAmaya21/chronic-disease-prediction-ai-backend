# Chronic Disease AI - Backend Setup

Este documento describe el proceso paso a paso para configurar y ejecutar el backend en Django.

## Requisitos Previos

- Python 3.8 o superior instalado
- pip (gestor de paquetes de Python)
- Git (opcional, si se clona el repositorio)

## Instalación y Configuración

### 1. Navegar al directorio del proyecto

```bash
cd chronic_disease_ai
```

### 2. Crear el entorno virtual

Es importante crear un entorno virtual para aislar las dependencias del proyecto:

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

**En Windows:**
```bash
venv\Scripts\activate
```

**En Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar las dependencias

Una vez activado el entorno virtual, instala todas las dependencias necesarias:

```bash
pip install -r requirements
```

### 5. Realizar las migraciones de la base de datos

Ejecuta las migraciones para crear las tablas necesarias en la base de datos:

```bash
python manage.py migrate
```

### 6. Ejecutar el servidor de desarrollo

Finalmente, inicia el servidor de desarrollo de Django:

```bash
python manage.py runserver
```