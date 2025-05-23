#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$whl" --plat "$PLAT" -w /wheelhouse/
        rm "$whl"
    fi
}

declare -a PythonVersions=("cp39-cp39" "cp310-cp310" "cp311-cp311" "cp312-cp312")

for val in ${PythonVersions[@]}; do
    /opt/python/$val/bin/pip install -r /src/requirements.txt
    /opt/python/$val/bin/pip wheel /src/ --no-deps -w /wheelhouse/
done

suffix=*_x86_64.whl

for whl in /wheelhouse/$suffix; do
    repair_wheel "$whl"
done
