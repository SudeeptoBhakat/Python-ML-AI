import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv
import joblib
import json
import numpy as np
import pywt
import cv2
load_dotenv()

def load_model_and_dict():
    model = joblib.load('img_classifier.pkl')
    with open("name_dict.json", "r") as f:
        class_dict = json.load(f)
    return model, class_dict

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def content_generator(query):
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        max_length=128,
        temperature=0.7,
        token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )

    examples = [
        {
            "query": "What is a mobile?",
            "answer": "A mobile is a portable communication device, commonly known as a mobile phone or cell phone. It allows users to make calls, send messages, access the internet, and use various applications."
        },
        {
            "query": "Why is the sky blue?",
            "answer": "The sky appears blue because molecules in the air scatter blue light from the sun more than they scatter red light."
        },
        {
            "query": "How do birds fly?",
            "answer": "Birds fly by using their wings, which provide lift and thrust."
        }
    ]

    example_template = """
    Question: {query}
    Response: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    suffix = """
    Question: {template_userInput}
    Response: """

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )

    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix=suffix,
        input_variables=["template_userInput", "template_tasktype_option"],
        example_separator="\n"
    )
    prompt = new_prompt_template.format(template_userInput=query)
    response = llm.invoke(prompt)
    return response