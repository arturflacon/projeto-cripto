```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
Remove-Item Env:USAR_SNAPSHOT -ErrorAction SilentlyContinue
python src/collect.py; python src/preprocess.py; python src/features.py
python src/clustering.py; python src/classification.py; python src/regression.py
python src/prever.py
python tests/test_vazamento.py
code reports/previsoes.md
```
