param([string]$Dataset = "$PSScriptRoot\covid-files\dataset.json")
python "$PSScriptRoot\pipeline\manage.py" status --dataset "$Dataset"
exit $LASTEXITCODE
