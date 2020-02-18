#!/bin/bash
# Downloads and extracts PyDesigner
mkdir -p /tmp/PyDesigner
export URL=$(curl -s https://api.github.com/repos/m-ama/PyDesigner/releases | jq -r '.[0] | .tarball_url')
wget -q -O - $URL | tar -xzvf - -C /tmp/PyDesigner --strip 1