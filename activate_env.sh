poetry config virtualenvs.in-project true
poetry lock
poetry install
source $(poetry env info --path)/bin/activate
echo "Agent Activated"
