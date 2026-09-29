# Sobe a busca e a página. Rode na pasta do projeto.
# Depois abra http://localhost:5173
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Error "Crie o ambiente primeiro: python -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
$jpegs = @(Get-ChildItem ".\data\raw\laion_*.jpg" -ErrorAction SilentlyContinue)
if ($jpegs.Count -lt 500) {
  Write-Error "Faltam as 500 imagens em data\raw. Elas nao vao no Git. Rode: python -m src.scraper.laion_downloader"
}

$env:PYTHONIOENCODING = "utf-8"
Start-Process -WindowStyle Minimized -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1", "--port", "8000"
if (-not (Test-Path ".\web\node_modules")) {
  npm --prefix web install
}
Start-Process -WindowStyle Minimized -FilePath "npm" -ArgumentList "--prefix", "web", "run", "dev"
Start-Sleep -Seconds 2
Start-Process "http://localhost:5173"
Write-Output "A busca sobe na porta 8000 (cerca de um minuto na primeira vez). A pagina e http://localhost:5173"
