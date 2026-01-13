# GdDS-Projekt2

## Aufgabe 2 
Um das Alpha Zero Framework mit dem TicTacToe aus dem AlphaZero Repo nutzen zu können, haben wir folgende Schritte in WSL2 verwendet.

### Repo klonen
`git clone https://github.com/suragnair/alpha-zero-general.git`

### Venv erstellen
Virtuelle Umgebung um Probleme mit den Dependencies lösen zu können

#### Venv herunterladen 
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```
#### Venv erstellen
```bash
python3 -m venv .venv
```

#### Venv aktivieren
```bash
source .venv/bin/activate
```

### Dependencies installieren

Diese Schritte wurden nur im Venv getestest!

#### Cuda Toolkit installieren
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
```
Quelle: [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

#### Tensor Flow installieren

```bash
 python3 -m pip install tensorflow==2.15.1
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
```
Quelle: [TensorFlow](https://www.tensorflow.org/install/pip#windows-wsl2)

#### Pip nachladen

```bash
pip install coloredlogs tqdm torch
```

### Code Anpassungen

#### main.py

```python3
from othello.OthelloGame import OthelloGame as Game                                                                            
from othello.pytorch.NNet import NNetWrapper as nn
```
 ->
```python3
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as nn
```
Für Game muss ein Alias angelegt werden, direkt nach den import:
```
Game = TicTacToeGame
``` 

#### 


### Ausführen
```bash
python main.py
