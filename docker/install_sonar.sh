#!/bin/bash -ex

wget -nv https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
unzip build-wrapper-linux-x86.zip
mv build-wrapper-linux-x86/* /usr/local/bin/
rm -rf build-wrapper-linux-x86*

wget -nv https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-3.3.0.1492-linux.zip
unzip sonar-scanner-cli-3.3.0.1492-linux.zip
mv sonar-scanner-3.3.0.1492-linux /opt/
ln -sf /opt/sonar-scanner-3.3.0.1492-linux /opt/sonar-scanner
rm -rf sonar-scanner-cli-3.3.0.1492-linux.zip

