import sys
from collections import defaultdict
from pathlib import Path
import pandas as df
import fitz 

#!pip install PyMuPDF


def getDataframe_txt(ruta):
    results = defaultdict(list)
    for file in Path(ruta).iterdir():
        with open(file, "r", encoding= 'utf-8') as file_open:
            results["document_title"].append(file.name)
            results["text"].append(file_open.read())
    pd = df.DataFrame(results)

    return pd



def getDataframe_pdf(ruta):
    results = defaultdict(list)
    for file in Path(ruta).iterdir():
        with fitz.open(file) as file_open:
            text = ""
            for page in file_open:
                text += page.getText()
            results["document_title"].append(file.name)
            results["text"].append(text)
    pd = df.DataFrame(results)

    return pd



