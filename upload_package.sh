#!/bin/bash
cd dist
rm *
cd ..
python3 -m build
python3 -m twine upload dist/*
