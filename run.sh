#!/bin/bash

# Configuración de directorios
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"
MODEL_DIR="$PROJECT_ROOT/save_models"  # Cambiado a save_models para coincidir con tu código

echo "🔧 Creando carpetas necesarias..."
mkdir -p "$BUILD_DIR"
mkdir -p "$MODEL_DIR"  # Asegura que save_models existe

echo "📁 Configurando el proyecto..."
cd "$BUILD_DIR"
cmake ..

echo "🛠️ Compilando con make..."
make

echo "🚀 Ejecutando el programa..."
# Ejecuta desde el directorio raíz para que las rutas relativas funcionen
cd "$PROJECT_ROOT"
"$BUILD_DIR"/test