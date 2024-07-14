# Usar una imagen base de Python 3.7.16
FROM python:3.7.16

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido del directorio actual al contenedor
COPY . .

EXPOSE 8000

# Comando por defecto para ejecutar el contenedor
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
