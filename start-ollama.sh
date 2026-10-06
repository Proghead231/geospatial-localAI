#!/bin/bash

export OLLAMA_MODELS="/run/media/proghead/Nice!/Software/.ollama/models"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

konsole --hold -e bash -c 'ollama serve'