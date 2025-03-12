# Test Deepskin

## Infos

All architecture ``schemas`` available [here](./SCHEMES.md)

* Install a root a folder ``Deepskin`` : <https://github.com/Nico-Curti/Deepskin>
* ``Notebooks`` for usage can be found here : <https://github.com/Nico-Curti/Deepskin/tree/main/docs/source/notebooks>
* Once the script used, the model can be found here : ``.venv\Lib\site-packages\checkpoints\efficientnetb3_deepskin_semantic.h5`` or ``.venv/lib/python3.x/site-packages/checkpoints/efficientnetb3_deepskin_semantic.h5``
* Or download it here : <https://drive.google.com/uc?id=1it-fXhSTFp49kS6I0_ceykZ8jqtYLPkL> (For own tests)
* Published in ``MDPI`` : <https://www.mdpi.com/1422-0067/24/1/706>

## Start

Be sure to be in the ``test_deepskin`` root directory.

## Init

```bash
bash init.sh
```

## Run

### Demo

### UI

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/Scripts/python -m demo.ui
```

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/bin/python3 -m demo.ui
```

### CLI

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/Scripts/python -m demo.cli
```

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/bin/python3 -m demo.cli
```

### API

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/Scripts/python -m api.main
```

```bash
export TF_ENABLE_ONEDNN_OPTS=0 && .venv/bin/python3 -m api.main
```

## Lint

```bash
.venv/Scripts/autopep8 --exclude='.venv' --in-place --recursive --aggressive .
```

```bash
.venv/bin/autopep8 --exclude='.venv' --in-place --recursive --aggressive .
```

## Clean

```bash
rm -rf Deepskin .venv && git clean -fdX
```

## Docker

### Build

```bash
docker compose up --build -d
```

### Remove

```bash
docker compose down -v --rmi all
```
