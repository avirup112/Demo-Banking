Write-Host "Fixing DVC conflicts and running dvc repro..." -ForegroundColor Green

Write-Host "Step 1: Removing conflicting directories from Git tracking..." -ForegroundColor Yellow
try { git rm -r --cached "models\artifacts" } catch { Write-Host "models\artifacts not in git" }
try { git rm -r --cached "models\trained" } catch { Write-Host "models\trained not in git" }
try { git rm -r --cached "data\processed" } catch { Write-Host "data\processed not in git" }
try { git rm -r --cached "explanations" } catch { Write-Host "explanations not in git" }
try { git rm -r --cached "reports\metrics.json" } catch { Write-Host "metrics.json not in git" }

Write-Host "Step 2: Committing Git changes..." -ForegroundColor Yellow
git add .
git commit -m "stop tracking DVC-managed directories"

Write-Host "Step 3: Updating .gitignore..." -ForegroundColor Yellow
Add-Content -Path ".gitignore" -Value @"

# DVC managed directories
/models/artifacts/
/models/trained/
/data/processed/
/explanations/
/reports/metrics.json
"@

Write-Host "Step 4: Committing .gitignore changes..." -ForegroundColor Yellow
git add .gitignore
git commit -m "update gitignore for DVC directories"

Write-Host "Step 5: Running DVC repro..." -ForegroundColor Green
dvc repro

Write-Host "DVC pipeline completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"