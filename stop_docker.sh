#!/bin/bash
docker stop $(docker ps -a -q) | xargs docker rm
