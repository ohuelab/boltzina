#!/bin/bash
# Install Boltz-2 model files
echo "Installing Boltz-2 model files..."
python setup_boltzina.py

# Install AutoDock Vina
mkdir -p bin
echo "Installing AutoDock Vina..."
if [ ! -f "bin/vina" ]; then
    wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 -O bin/vina
fi

chmod +x bin/vina


# Install MAXIT
echo "Installing MAXIT..."
if [ ! -f "maxit-v11.300-prod-src.tar.gz" ]; then
    wget https://sw-tools.rcsb.org/apps/MAXIT/maxit-v11.300-prod-src.tar.gz
fi
if [ ! -d "maxit-v11.300-prod-src" ]; then
    tar xvf maxit-v11.300-prod-src.tar.gz
fi

if [ ! -f "maxit-v11.300-prod-src/bin/maxit" ]; then
    cd maxit-v11.300-prod-src
    make binary
    cd ..
fi

# mv maxit-v11.300-prod-src/bin/maxit bin/

echo "Installation complete!"
echo "========================================================"
echo "Add the following lines to your ~/.bashrc or ~/.zshrc:"
echo "export RCSBROOT=`realpath maxit-v11.300-prod-src`"
echo "export PATH=\$PATH:\$RCSBROOT/bin"
echo "export PATH=\$PATH:`realpath bin`"
