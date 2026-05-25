# -*- coding: utf-8 -*-
from pydoc import cli
import requests
import json
import sys
import io
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
import os


