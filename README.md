```console
curl -LsSf https://astral.sh/uv/install.sh | sh
. ~/.bashrc
uv sync
uv run main.py
# uv run --with jupyter jupyter lab --allow-root --port 8080 --NotebookApp.token='' --NotebookApp.password=''
# uv run python -m ipykernel install --user --name=train
```