#!/bin/bash

echo "📦 Creating virtual environment in .venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Setup complete. Activate with: source .venv/bin/activate"
