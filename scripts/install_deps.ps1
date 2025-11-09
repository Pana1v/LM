# PowerShell script to install Eigen and CUDA dependencies
# Run: .\scripts\install_deps.ps1

Write-Host "=== Installing Dependencies for Levenberg-Marquardt Project ===" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Function to check if command exists
function Test-Command {
    param($cmd)
    $null = Get-Command $cmd -ErrorAction SilentlyContinue
    return $?
}

# Install Eigen3
Write-Host "`n[1/2] Installing Eigen3..." -ForegroundColor Yellow

$eigenInstalled = $false

# Method 1: Check if vcpkg is available
if (Test-Command vcpkg) {
    Write-Host "  Found vcpkg, installing Eigen3 via vcpkg..." -ForegroundColor Green
    try {
        vcpkg install eigen3:x64-windows
        if ($LASTEXITCODE -eq 0) {
            $eigenInstalled = $true
            Write-Host "  + Eigen3 installed via vcpkg" -ForegroundColor Green
        }
    } catch {
        Write-Host "  X vcpkg installation failed" -ForegroundColor Red
    }
}

# Method 2: Check if Chocolatey is available
if (-not $eigenInstalled -and (Test-Command choco)) {
    Write-Host "  Found Chocolatey, installing Eigen3 via Chocolatey..." -ForegroundColor Green
    if ($isAdmin) {
        try {
            choco install eigen --version=3.4.0 -y
            if ($LASTEXITCODE -eq 0) {
                $eigenInstalled = $true
                Write-Host "  + Eigen3 installed via Chocolatey" -ForegroundColor Green
            }
        } catch {
            Write-Host "  X Chocolatey installation failed" -ForegroundColor Red
        }
    } else {
        Write-Host "  ! Administrator rights required for Chocolatey" -ForegroundColor Yellow
    }
}

# Method 3: Manual download and extract
if (-not $eigenInstalled) {
    Write-Host "  Downloading Eigen3 manually..." -ForegroundColor Yellow
    $eigenUrl = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    $eigenDir = "C:\Eigen3"
    $eigenZip = "$env:TEMP\eigen-3.4.0.zip"
    
    if (-not (Test-Path $eigenDir)) {
        try {
            Write-Host "  Downloading from $eigenUrl..." -ForegroundColor Gray
            Invoke-WebRequest -Uri $eigenUrl -OutFile $eigenZip -UseBasicParsing
            
            Write-Host "  Extracting to $eigenDir..." -ForegroundColor Gray
            Expand-Archive -Path $eigenZip -DestinationPath "$env:TEMP" -Force
            $extracted = "$env:TEMP\eigen-3.4.0"
            
            if (Test-Path $extracted) {
                New-Item -ItemType Directory -Path $eigenDir -Force | Out-Null
                Move-Item -Path "$extracted\*" -Destination $eigenDir -Force
                Remove-Item -Path $extracted -Recurse -Force
                Remove-Item -Path $eigenZip -Force
                
                $eigenInstalled = $true
                Write-Host "  + Eigen3 extracted to $eigenDir" -ForegroundColor Green
                Write-Host "  Set environment variable: `$env:EIGEN3_ROOT = '$eigenDir'" -ForegroundColor Cyan
            }
        } catch {
            Write-Host "  X Download/extraction failed: $_" -ForegroundColor Red
            Write-Host "  Please download manually from: https://eigen.tuxfamily.org/" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  + Eigen3 already exists at $eigenDir" -ForegroundColor Green
        $eigenInstalled = $true
    }
}

# Verify Eigen installation
if ($eigenInstalled) {
    $eigenPath = if (Test-Path "C:\Eigen3\Eigen") { "C:\Eigen3" }
                 elseif ($env:EIGEN3_ROOT) { $env:EIGEN3_ROOT }
                 elseif ($env:VCPKG_ROOT) { "$env:VCPKG_ROOT\installed\x64-windows\include\eigen3" }
                 else { $null }
    
    if ($eigenPath -and (Test-Path "$eigenPath\Eigen")) {
        Write-Host "  + Eigen3 verified at: $eigenPath" -ForegroundColor Green
    }
}

# Install CUDA
Write-Host "`n[2/2] Checking CUDA..." -ForegroundColor Yellow

$cudaInstalled = $false

# Check if CUDA is already installed
if (Test-Command nvcc) {
    $cudaVersion = nvcc --version 2>&1 | Select-String "release" | ForEach-Object { $_.Line }
    Write-Host "  + CUDA already installed: $cudaVersion" -ForegroundColor Green
    $cudaInstalled = $true
} elseif (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
    $cudaVersions = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory
    if ($cudaVersions) {
        $latest = $cudaVersions | Sort-Object Name -Descending | Select-Object -First 1
        Write-Host "  + CUDA found at: $($latest.FullName)" -ForegroundColor Green
        Write-Host "  Adding to PATH..." -ForegroundColor Gray
        $env:PATH += ";$($latest.FullName)\bin"
        $cudaInstalled = $true
    }
}

# Check for NVIDIA GPU
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
if (-not $gpu) {
    Write-Host "  ! No NVIDIA GPU detected. CUDA requires an NVIDIA GPU." -ForegroundColor Yellow
    Write-Host "  CUDA will not work on this system." -ForegroundColor Yellow
} else {
    Write-Host "  + NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green
}

if (-not $cudaInstalled) {
    Write-Host "  CUDA not installed. Installation options:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
    Write-Host "  2. Or use Chocolatey (if admin): choco install cuda -y" -ForegroundColor Cyan
    
    if ($isAdmin -and (Test-Command choco)) {
        $install = Read-Host "  Install CUDA via Chocolatey? (y/n)"
        if ($install -eq 'y') {
            try {
                choco install cuda -y
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  + CUDA installed via Chocolatey" -ForegroundColor Green
                    $cudaInstalled = $true
                }
            } catch {
                Write-Host "  X CUDA installation failed" -ForegroundColor Red
            }
        }
    }
}

# Summary
Write-Host "`n=== Installation Summary ===" -ForegroundColor Cyan
Write-Host "Eigen3: $(if ($eigenInstalled) { '+ Installed' } else { 'X Not installed' })" -ForegroundColor $(if ($eigenInstalled) { 'Green' } else { 'Red' })
Write-Host "CUDA:   $(if ($cudaInstalled) { '+ Installed' } else { 'X Not installed' })" -ForegroundColor $(if ($cudaInstalled) { 'Green' } else { 'Red' })

if ($eigenInstalled -and $cudaInstalled) {
    Write-Host "`n+ All dependencies installed! You can now build all implementations." -ForegroundColor Green
} else {
    Write-Host "`n! Some dependencies missing. See INSTALL_EIGEN_CUDA.md for manual installation." -ForegroundColor Yellow
}

