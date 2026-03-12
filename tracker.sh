#!/bin/bash
uri=127.0.0.1
port=08254
runs=file:///home/stef/cgr/artifacts/mlruns
mlflow ui --host $uri --port $port --backend-store-uri $runs