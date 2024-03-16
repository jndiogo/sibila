./cleanup.sh

cp models/_original_models.json models/models.json
cp models/_original_formats.json models/formats.json

./run_mypy.sh
./run_flake8.sh

./build_docs.sh
mkdocs build

tests/setup.sh
pytest
